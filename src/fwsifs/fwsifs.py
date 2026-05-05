import numpy as np
import time
from scipy import ndimage

from . import nlls, pmath, romeo


NAX = np.newaxis
GAMMA_1H = 42.58e3 # kHz/T
CS_WATER = 4.7 # ppm
WATER_T2 = 35 # ms



VARIABLES = ['b0', 'r2', 'wchi', 'r2w', 'r2f', 'ndb']

def pdff(
        echo_times, volumes, *,
        mask=None, 
        fat_model='Bydder2011', # 'Reyngoudt2024',
        variables=['b0', 'r2', 'wchi'], # r2w, r2f, ndb
        coarse=False,
        disp=True, 
        plot=False,
        field_strength=3, 
        pixel_spacing=None,
        return_fat_model=False, 
        niter_init=10, niter_b0=100, niter_refine=50,
        filter_phi0=True, phi0_filter_size=1, 
        init_r2=0, mu_r2=None, 
        b0_filter_size=(3, 3, 5), mu_b0=None,
        chi_filter_size=1, mu_chi=1e3, init_fatppm=2,
        r2w_filter_size=1, mu_r2w=1e3, t2w=40, r2wpow=1, #t2w=25
        r2f_filter_size=1, mu_r2f=5e3, t2f=80, r2fpow=2,
        ndb_filter_size=1, mu_ndb=1e2,

    ):
    """ fat/water separation method """
    tic = time.time()
    info = {
        'echo_times': echo_times,
        'variables': variables,
        'fat_model': fat_model,
        'coarse': coarse,
        'field_strength': field_strength,
        'pixel_spacing': pixel_spacing,
        'niter_init': niter_init,
        'niter_b0': niter_b0,
        'niter_refine': niter_refine,
        'mu_b0': mu_b0,
        'b0_filter_size': b0_filter_size,
        'filter_phi0': filter_phi0,
        'phi0_filter_size': phi0_filter_size,
        'init_r2': init_r2,
        'mu_r2': mu_r2,
        'init_fatppm': init_fatppm, 
        'mu_chi': mu_chi,
        'chi_filter_size': chi_filter_size,
        't2w': t2w,
        'r2wpow': r2wpow,
        'mu_r2w': mu_r2w,
        'r2w_filter_size': r2w_filter_size,
        't2f': t2f,
        'r2fpow': r2fpow,
        'mu_r2f': mu_r2f,
        'r2f_filter_size': r2f_filter_size,
        'mu_ndb': mu_ndb,
        'ndb_filter_size': ndb_filter_size,
    }

    # check fit parameters
    invalid = set(variables) - set(VARIABLES)
    if invalid:
        raise ValueError(f'Unknown variable: {", ".join(invalid)}')
    variables = [var for var in VARIABLES if var in variables]

    # setup
    if mask is None:
        mask = np.ones_like(volumes[0], dtype=bool)
    npix = mask.sum()
    times = np.asarray(echo_times) 
    spacing = np.array([1] * mask.ndim if pixel_spacing is None else pixel_spacing)
    obs = np.stack([vol[mask] for vol in volumes], axis=-1)
    norm = np.linalg.norm(obs, axis=-1)
    ignore = norm < 1e-8    
    
    # setup fat model
    if fat_model == 'Reyngoudt2024':
        fat_model = FatModelNDB(**MODEL_REYNGOUDT2024, fs=field_strength)
    elif fat_model == 'Bydder2011':
        fat_model = FatModelNDB(**MODEL_BYDDER2011, fs=field_strength)
    elif fat_model == 'Ren2008':
        fat_model = FatModel(**MODEL_REN2008, fs=field_strength)
    elif fat_model == 'Yu2008':
        fat_model = FatModel(**MODEL_YU2008, fs=field_strength)
    elif fat_model == 'Azzabou2017':
        fat_model = FatModel(**MODEL_AZZABOU2017, fs=field_strength)
    elif not isinstance(fat_model, FatModel):
        raise ValueError(f'Expecting FatModel object, not: {fat_model}')
    elif 'ndb' in variables and not isinstance(fat_model, FatModelNDB):
        raise ValueError(f'Expecting FatModelNDB object, not: {fat_model}')
                            
    # init R2_chi
    r2 = init_r2 * 1e-3 * np.ones((2, npix))
    r2pow = r2fpow # power law for global R2 
    # r2pow = r2wpow # power law for global R2 

    # init fat/water signal with R2 decay
    water = np.exp(-times / t2w)
    if np.allclose(fat_model.r2, 0):
        fat_model.r2 = 1 / t2f
    fat = fat_model(times)

    # init omega_chi (shift average fat frequency to match init_fatppm)
    chiinit = 0
    if 'wchi' in variables:
        mean_freq = np.sum(fat_model.freqs * fat_model.ampls)
        init_chippm = init_fatppm - FatModel.to_ppm(field_strength, mean_freq)
        chiinit = GAMMA_1H * field_strength * init_chippm * 1e-6 * 2 * np.pi # rad/ms
    wchi = chiinit

    # mixing matrix
    Afat = a_matrix(times, water, fat, wchi=wchi)

    # init with swapped candidates
    wf = np.ones((2, npix, 2))
    wf[0] = [0.8, 0.2]
    wf[1] = [0.2, 0.8]
    
    if disp: print('Init W and F')
    for i in range(niter_init):
        y = matvec(Afat, wf)
        # angle candidates
        phi = np.angle(y)
        # if coarse_r2 and (i > 0):
        if i > 0 and {'r2', 'r2w', 'r2f'} & set(variables):
            # estimate r2
            p = abs(y) * np.exp(- r2[..., NAX] * times**r2pow)
            p[p < 1e-8] = 1e-8
            # dr2 = np.mean((abs(obs) - p) / abs(times * p), axis=-1)
            dr2 = 0.3 * vecvec(p * times**r2pow, p - abs(obs)) / np.sum((times**r2pow * p)**2, axis=-1)
            r2 = np.clip(r2 + dr2, 0, 0.2)
        b = abs(obs) * np.exp(1j * phi + r2[..., NAX] * times**r2pow)
        # estimate w, f (magnitude fitting)
        wf, r = solve_nnls(Afat, b)
        resid = np.linalg.norm(r, axis=-1)
        if disp: print(f'iteration: {i + 1}, r={resid.sum(-1)}')
        if i >= 0 and 'wchi' in variables:
            # estimate wchi
            dchi = np.mean(np.imag((b - y) / fat / times / np.exp(1j * wchi[..., NAX] * times)), axis=-1)
            # dchi = np.imag(vecvec(np.conj(fat * times * np.exp(1j * wchi[..., NAX] * times)), b - y)) / np.sum(abs(fat * times)**2)
            dchi[0] /= np.clip(wf[0, ..., 1], 1, None)
            dchi[1] /= np.clip(wf[1, ..., 0], 1, None)
            wchi += np.clip(dchi, -1e-1, 1e-1)
            Afat = a_matrix(times, water, fat, wchi=wchi)
        
    # compare residuals
    sel = resid[0] < resid[1]
    psi = np.exp(1j * phi)
    
    # initialize b0 
    # (assumes regular echo times)
    delta_t = times[1] - times[0]
    
    psi_1 = np.sum(obs[:, 1:] * psi[0, :, :-1] * np.conj(obs[:, :-1] * psi[0, :, 1:]), axis=-1)
    psi_2 = np.sum(obs[:, 1:] * psi[1, :, :-1] * np.conj(obs[:, :-1] * psi[1, :, 1:]), axis=-1)
    psi_1 = np.exp(1j * np.angle(psi_1))
    psi_2 = np.exp(1j * np.angle(psi_2))

    # subtract susceptibility phase shift from total phase
    psi_1x, psi_2x = 1, 1
    if 'wchi' in variables:
        tot = np.sum(abs(wf), axis=-1)
        ff_1 = abs(wf[0, ..., 1]) / np.maximum(tot[0], 1)
        psi_1x = np.exp(1j * ff_1 * wchi[0] * delta_t)
        ff_2 = abs(wf[1, ..., 1]) / np.maximum(tot[1], 1)
        psi_2x = np.exp(1j * ff_2 * wchi[1] * delta_t)
        psi_1 *= psi_1x
        psi_2 *= psi_2x
    
    # confidence map from residuals
    conf = 1 - np.clip(np.min(resid, axis=0) / (norm + ignore), 0, 1)
    conf[ignore] = 0
    
    # select best b0 candidate
    if disp: print('Initialize B0.')
    filter_size = to_filter_size(b0_filter_size, spacing)
    for i in range(niter_b0):
        psi = psi_1 * sel + psi_2 * (1 - sel)
        psif = weighted_filter(psi, mask, conf=conf, filter_size=filter_size)
        psif = np.exp(1j * np.angle(psif))
        diff = [abs(psi_1 - psif),  abs(psi_2 - psif)]
        _sel = diff[0] < diff[1]
        nmoving = np.sum(_sel ^ sel)
        if disp: print(f'iter: {i + 1}, n. moving={nmoving}')
        if nmoving < 10:
            # decrease filter size
            filter_size = np.maximum(filter_size - 2, 1)
            if np.all(filter_size <= 1):
                break
        sel = _sel

    # add back phase shift
    psi = psif / (psi_1x * sel + psi_2x * (1 - sel))
    phi = np.angle(psi)

    # unwrap phi
    mag = np.mean(np.abs(volumes), axis=0)
    phi = romeo.unwrap(tovolume(phi, mask), mask=mask, mag=mag)[mask]    
    b0 = phi / delta_t
    
    # select r2 and wchi
    r2 = r2[0] * sel + r2[1] * (1 - sel)
    if 'wchi' in variables:
        wchi = wchi[0] * sel + wchi[1] * (1 - sel)

    if coarse:
        # coarse estimation from above estimates

        filter_size = to_filter_size(phi0_filter_size, spacing)
        def p_filter(p):
            pf = weighted_filter(p, mask, conf=conf, filter_size=filter_size)
            return p * conf + pf * (1 - conf)
        
        W = w_matrix(b0, r2, times, r2pow=r2pow)
        A = a_matrix(times, water, fat, wchi=wchi)
        wf, phi0, resid = solve_pcls(W[..., NAX] * A, obs, p_filter=p_filter)

    else:  
        # non linear least squares optimization
        r2w = r2 * times[-1]**(r2pow - r2wpow)
        r2f = r2 * times[-1]**(r2pow - r2fpow)
        if {'r2', 'r2w', 'r2f'} < set(variables):
            r2 = 0
        
        # init variables
        init = dict(
            b0=b0 if 'b0' in variables else 0,
            wchi=wchi if 'wchi' in variables else 0, 
            r2=r2 if 'r2' in variables else 0,
            r2w=r2w if 'r2w' in variables else 0,
            r2f=r2f if 'r2f' in variables else 0,
            ndb=fat_model.ndb if 'ndb' in variables else None,
        )

        # fit B0, R2, wchi with var pro for wf and phi0
        obj = Objective(
            variables, 
            dict(
                times=times, obs=obs, 
                mask=mask, 
                water=water, fat=fat, 
                r2pow=r2pow, r2wpow=r2wpow, r2fpow=r2fpow,
                fat_freqs=fat_model.freqs, 
                **init,
            ),
            options=dict(
                filter_phi0=filter_phi0, phi0_filter_size=phi0_filter_size,
                mu_r2=mu_r2,
                mu_b0=mu_b0, b0_filter_size=to_filter_size(b0_filter_size, spacing),
                mu_chi=mu_chi, chi_filter_size=to_filter_size(chi_filter_size, spacing),
                mu_r2w=mu_r2w, r2w_filter_size=to_filter_size(r2w_filter_size, spacing),
                mu_r2f=mu_r2f, r2f_filter_size=to_filter_size(r2f_filter_size, spacing),
                mu_ndb=mu_ndb, ndb_filter_size=to_filter_size(ndb_filter_size, spacing), 
                plot=plot,
            ),
        )
        # optimize
        res = nlls.nlls(obj, method='lm', disp=disp, maxiter=niter_refine) 

        # recover fitted values
        b0, r2, wchi, r2w, r2f, ndb = obj.getvars(res.x)
        wf, phi0 = obj.memory['wf'], obj.memory['phi0']
        resid = obj.resid(res.x)

        info['nlls.method'] = res.method
        info['nlls.message'] = res.msg
        info['nlls.cost'] = res.cost

    # ff value
    ignore = np.max(wf, axis=-1) < 1e-8
    ff = abs(wf[:, 1]) / (abs(wf[:, 0]) + abs(wf[:, 1]) + ignore)

    # fix values
    phi0 += np.pi * (np.sum(wf, axis=1) < 0)
    phi0 = np.mod(phi0 + np.pi, 2 * np.pi) - np.pi        
    if 'wchi' in variables:
        # remove water susceptibility shift from b0
        wchiw = - ff * wchi
        b0 = b0 - wchiw
    if 'ndb' in variables:
        ndb = np.clip(0, ndb, None)

    # normalized RMS deviation
    rmsd = np.linalg.norm(resid, axis=-1)
    nrmsd = rmsd / (norm + 1e-8 * ignore)

    # goodness of fit
    n, k = len(times), len(variables)
    bic = np.maximum(n * np.log(np.maximum(rmsd**2 / n, 1e-10)) + k * np.log(n), 0)

    if disp: print('Done.')
    info['computation_time'] = time.time() - tic
        
    # outputs
    volumes = {
        'mask': mask, # bool
        'ffmap': tovolume(ff, mask), # fraction
        'wmap': tovolume(abs(wf)[..., 0], mask), # amplitude
        'fmap': tovolume(abs(wf)[..., 1], mask), # amplitue
        'b0map': tovolume(b0, mask), # rad/ms
        'phi0': tovolume(phi0, mask), # rad
        'resids': tovolume(nrmsd, mask), # fraction
        'bic':  tovolume(bic, mask), 
    }
    if 'wchi' in variables:
        wchi = wchi / (2 * np.pi * GAMMA_1H * field_strength * 1e-6) # rad/ms to ppm
        volumes['wchi'] = tovolume(wchi, mask) # ppm
    if 'r2' in variables:
        volumes['r2'] = tovolume(r2 * 1e3**r2pow, mask) # 1/s
    if 'r2w' in variables:
        volumes['r2w'] = tovolume(r2w * 1e3**r2wpow, mask) # 1/s^r2wpow
        volumes['t2w'] = tovolume(1 / np.clip(1 / t2w + r2w * times[-1]**(r2wpow - 1), 1e-2, 1e5), mask) # ms
    if 'r2f' in variables:
        volumes['r2f'] = tovolume(r2f * 1e3**r2fpow, mask)# 1/s^r2fpow
    if 'ndb' in variables:
        volumes['ndb'] = tovolume(ndb, mask)

    if return_fat_model:
        return volumes, info, fat_model
    return volumes, info


#
# signal model

def signal(W, A, wf, phi0):
    """ signal model """
    return W * matvec(A, wf * np.exp(1j * phi0[:, NAX]))

def w_matrix(b0, r2, times, r2pow=1):
    """ B0/R2' evolution matrix """
    b0 = np.atleast_1d(b0)[..., NAX] if np.ndim(b0) else b0
    r2 = np.atleast_1d(r2)[..., NAX] if np.ndim(r2) else r2
    return pmath.evaluate('exp(1j * b0 * times - r2 * times**r2pow)', b0=b0, times=times, r2=r2, r2pow=r2pow)

def a_matrix(times, water, fat, *, wchi=0, r2w=0, r2f=0, r2wpow=1, r2fpow=1):
    """ Water/Fat evolution matrix"""
    wchi = np.atleast_1d(wchi)[..., NAX] if np.ndim(wchi) else wchi
    r2w = np.atleast_1d(r2w)[..., NAX] if np.ndim(r2w) else r2w
    r2f = np.atleast_1d(r2f)[..., NAX] if np.ndim(r2f) else r2f
    water = pmath.evaluate('water * exp(-r2w * times**r2wpow)', water=water, r2w=r2w, times=times, r2wpow=r2wpow)
    fat = pmath.evaluate('fat * exp(1j * wchi * times - r2f * times**r2fpow)', fat=fat, wchi=wchi, times=times, r2f=r2f, r2fpow=r2fpow)
    return np.stack(np.broadcast_arrays(water, fat), axis=-1)


#
# objective function

class Objective(nlls.Objective):
    """ PDFF objective """

    def __init__(self, variables, parameters, options=None):
        self.variables = variables
        self.params = parameters
        self.opts = options or {}
        self.memory = {}
        self.iter = 0
        self.opts['regularize_b0'] = ('b0' in variables) and bool(self.opts.get('mu_b0'))
        self.opts['regularize_chi'] = ('wchi' in variables) and bool(self.opts.get('mu_chi'))
        self.opts['regularize_r2w'] = ('r2w' in variables) and bool(self.opts.get('mu_r2w'))
        self.opts['regularize_r2f'] = ('r2f' in variables) and bool(self.opts.get('mu_r2f'))
        self.opts['regularize_ndb'] = ('ndb' in variables) and bool(self.opts.get('mu_ndb'))


    def init(self):
        vars_ = [self.params[name] for name in VARIABLES if name in self.variables]
        shape = next(iter(var.shape for var in vars_ if np.size(var) > 1))
        return np.stack([np.broadcast_to(var, shape) for var in vars_], axis=1)

    def getvars(self, x):
        vars_ = []
        idx = 0
        for name in VARIABLES:
            if name in self.variables:
                vars_.append(x[:, idx])
                idx += 1
            else:
                vars_.append(self.params[name])
        return tuple(vars_)
    
    def setvar(self, x, name, value):
        idx = self.variables.index(name)
        x[:, idx] = value
    
    def update(self, x):
        b0, r2, wchi, r2w, r2f, ndb = self.getvars(x)

        T, b = self.params['times'], self.params['obs']
        mask = self.params['mask']
        freqs = self.params['fat_freqs']

        # fat signal
        water = self.params['water']
        fat = ndb_fat_model(ndb, freqs, T) if ndb is not None else self.params['fat']

        # fix bad values
        if 'ndb' in self.variables:
            ndb = np.clip(ndb, 0, 5)
            self.setvar(x, 'ndb', ndb)

        # solve pcls
        r2pow, r2wpow, r2fpow = self.params['r2pow'], self.params['r2wpow'], self.params['r2fpow']
        W = w_matrix(b0, r2, T, r2pow=r2pow)
        A = a_matrix(T, water, fat, wchi=wchi, r2w=r2w, r2f=r2f, r2wpow=r2wpow, r2fpow=r2fpow)
        AW = W[..., NAX] * A

        # M = np.real(hermitian(AW) @ AW)
        M = np.real(matmat(hermitian(AW), AW))
        iM = np.linalg.pinv(M)
        AWb = matvec(hermitian(AW), b)
        z = matvec(iM, AWb) 
        p = vecvec(AWb, z)
        
        # filter phi0
        if self.opts.get('filter_phi0', False):
            filter_size = self.opts['phi0_filter_size']
            conf = self.memory.get('conf', 1)
            pf = weighted_filter(p, mask, conf=conf, filter_size=filter_size)
            p = p * conf + pf * (1 - conf)

        # phi0 and w/f
        phi0 = 0.5 * np.angle(p)
        wf = np.real(z * np.exp(-1j * phi0)[..., NAX])
        ff = abs(wf[:, 1]) / np.clip(np.sum(abs(wf), axis=1), 1e-8, None)
        conf = 4 * ff * (1 - ff)

        # store variables
        memory = dict(wf=wf, phi0=phi0, M=M, iM=iM, z=z, p=p, conf=conf, ff=ff)

        # regularization
        if self.opts['regularize_b0']:
            # subtract water susceptibility shift from B0 before filtering
            wchiw = - wchi * np.clip(ff, 0, 1)
            fsize = self.opts['b0_filter_size']
            b0f = weighted_filter(b0 - wchiw, mask, conf=conf, filter_size=fsize)
            memory['b0f'] = b0f + wchiw
        if self.opts['regularize_chi']:   
            fsize = self.opts['chi_filter_size']
            memory['wchi_f'] = weighted_filter(wchi, mask, conf=conf, filter_size=fsize)
        if self.opts['regularize_r2w']:   
            fsize = self.opts['r2w_filter_size']
            memory['r2w_f'] = weighted_filter(r2w, mask, conf=1 - ff, filter_size=fsize)
        if self.opts['regularize_r2f']:   
            fsize = self.opts['r2f_filter_size']
            memory['r2f_f'] = weighted_filter(r2f, mask, conf=ff, filter_size=fsize)
        if self.opts['regularize_ndb']:   
            fsize = self.opts['ndb_filter_size']
            guide = abs(wf[:, 1])
            memory['ndbf'] = guided_filter(ndb, mask, guide=guide, filter_size=fsize)

        # plot
        # if self.opts.get('plot', True):
        #     # residual
        #     pred = W * matvec(A, wf * np.exp(1j * phi0[:, NAX]))
        #     rmsd = np.linalg.norm(pred - b, axis=-1)
        #     plot_optimization(
        #         self, 
        #         dict(wf=wf, phi0=phi0, b0=b0, r2=r2, wchi=wchi, r2w=r2w, r2f=r2f, ndb=ndb, rmsd=rmsd, pred=pred),
        #     )

        self.iter += 1
        self.memory.update(memory)
        return self.fun(x)

    def resid(self, x):
        """ signal residual """
        b0, r2, wchi, r2w, r2f, ndb = self.getvars(x)

        wf, phi0 = self.memory['wf'], self.memory['phi0']
        T, b = self.params['times'], self.params['obs']
        freqs = self.params['fat_freqs']

        # fat signal
        water = self.params['water']
        fat = ndb_fat_model(ndb, freqs, T) if ndb is not None else self.params['fat']

        # residual
        r2pow, r2wpow, r2fpow = self.params['r2pow'], self.params['r2wpow'], self.params['r2fpow']
        W = w_matrix(b0, r2, T, r2pow=r2pow)
        A = a_matrix(T, water, fat, wchi=wchi, r2w=r2w, r2f=r2f, r2wpow=r2wpow, r2fpow=r2fpow)
        return signal(W, A, wf, phi0) - b
        
    def fun(self, x):
        """ total residual """
        # residuals and regularization term
        f = self.resid(x)
        b0, r2, wchi, r2w, r2f, ndb = self.getvars(x)

        conf, ff = self.memory['conf'], self.memory['ff']
        for var in self.variables:
            if var== 'b0' and self.opts['regularize_b0']:
                f = np.pad(f, [(0, 0), (0, 1)])
                b0f = self.memory['b0f']
                f[:, -1] = self.opts['mu_b0'] * (b0 - b0f)
            if var == 'wchi' and self.opts['regularize_chi']:
                f = np.pad(f, [(0, 0), (0, 1)])
                wchi_f = self.memory['wchi_f']
                f[:, -1] = self.opts['mu_chi'] * (1 - conf) * (wchi - wchi_f)
            if var == 'r2w' and self.opts['regularize_r2w']:
                f = np.pad(f, [(0, 0), (0, 1)])
                r2w_f = self.memory['r2w_f']
                f[:, -1] = self.opts['mu_r2w'] * ff * (r2w - r2w_f)
            if var == 'r2f' and self.opts['regularize_r2f']:
                f = np.pad(f, [(0, 0), (0, 1)])
                r2f_f = self.memory['r2f_f']
                f[:, -1] = self.opts['mu_r2f'] * (1 - ff) * (r2f - r2f_f)
            if var == 'ndb' and self.opts['regularize_ndb']:
                f = np.pad(f, [(0, 0), (0, 1)])
                ndbf = self.memory['ndbf']
                f[:, -1] = self.opts['mu_ndb'] * (ndb - ndbf)
        return f

    def _jac(self, x):
        """ objective partial derivatives (jacobian) """
        b0, r2, wchi, r2w, r2f, ndb = self.getvars(x)

        T, b = self.params['times'], self.params['obs']
        wf, phi0 = self.memory['wf'], self.memory['phi0']
        p, z = self.memory['p'], self.memory['z']
        M, iM = self.memory['M'], self.memory['iM']
        freqs = self.params['fat_freqs']

        # fat signal
        water = self.params['water']
        fat = ndb_fat_model(ndb, freqs, T) if ndb is not None else self.params['fat']

        # signal
        r2pow, r2wpow, r2fpow = self.params['r2pow'], self.params['r2wpow'], self.params['r2fpow']
        W = w_matrix(b0, r2, T, r2pow=r2pow)
        A = a_matrix(T, water, fat, wchi=wchi, r2w=r2w, r2f=r2f, r2wpow=r2wpow, r2fpow=r2fpow)
        sig = signal(W, A, wf, phi0)

        AW = W[..., NAX] * A
        AWb = matvec(hermitian(AW), b)
        AWTb = matvec(hermitian(AW), T * b)
        P0 = np.exp(-1j * phi0)[:, NAX]
        ignore = abs(p) < 1e8
        y = matvec(iM, AWTb) 
        q = vecvec(AWb, y)

        jac = []
        for var in self.variables:
            if var == 'b0':
                # signal gradient wrt b0
                dphi0 = - np.real(q / (p + ignore))
                dwf = np.imag((y + z * dphi0[:, NAX]) * P0)
                g_b0 = signal(1j * W * T, A, wf, phi0)
                g_b0 += signal(W, A, dwf, phi0)
                g_b0 += sig * 1j * dphi0[:, NAX]
                jac.append(g_b0)

            if var == 'r2': 
                # signal gradient wrt R2
                T_ = T**r2pow
                AWTb_ = matvec(hermitian(AW), T_ * b) if r2pow != 1 else AWTb
                y_ = matvec(iM, AWTb_) if r2pow != 1 else y
                q_ = vecvec(AWb, y_) if r2pow != 1 else q
                # H_ = np.real((hermitian(AW) * T_) @ AW)
                H_ = np.real(matmat(AW.conj().mT * T_, AW))
                s_ = vecvec(matvec(H_, z), z)

                dphi0 = np.imag((s_ - q_) / (p + ignore))
                dwf = - np.real((y_ + 1j * z * dphi0[:, NAX]) * P0)
                dwf += 2 * matvec(iM, matvec(H_, wf))
                dsig = signal(- W * T_, A, wf, phi0)
                dsig += signal(W, A, dwf, phi0)
                dsig += sig * 1j * dphi0[:, NAX]
                jac.append(dsig)

            if var == 'wchi':
                # signal gradient wrt wchi
                dA = A * np.array([0, 1])
                y_ = vecvec(np.conj(A[..., 1] * W * T), b)[:, NAX] * iM[..., 1]
                q_ = vecvec(AWb, y_)
                # H_ = 1/2 * np.real(1j * (hermitian(AW) * T) @ (W[..., NAX] * dA))
                H_ = 1/2 * np.real(1j * matmat(AW.conj().mT * T, W[..., NAX] * dA))
                H_ += hermitian(H_)
                s_ = vecvec(matvec(H_, z), z)

                dphi0 = - np.imag((s_ + 1j * q_) / (p + ignore))
                dwf = np.real((-1j * y_ - 1j * z * dphi0[:, NAX]) * P0)
                dwf -= 2 * matvec(iM, matvec(H_, wf))
                dsig = signal(W, 1j * T[:, NAX] * dA, wf, phi0)
                dsig += signal(W, A, dwf, phi0)
                dsig += sig * 1j * dphi0[:, NAX]
                jac.append(dsig)

            if var == 'r2w':
                # signal gradient wrt r2w
                T_ = T**r2wpow 
                dA = A * np.array([1, 0])
                y_ = vecvec(np.conj(A[...,0] * W * T_), b)[:, NAX] * iM[..., 0]
                q_ = vecvec(AWb, y_)
                # H_ = 1/2 * np.real((hermitian(AW) * T_) @ (W[..., NAX] * dA))
                H_ = 1/2 * np.real(matmat(AW.conj().mT * T_, W[..., NAX] * dA))
                H_ += hermitian(H_)
                s_ = vecvec(matvec(H_, z), z)

                dphi0 = np.imag((s_ - q_) / (p + ignore))
                dwf = np.real((-y_ - 1j * z * dphi0[:, NAX]) * P0)
                dwf += 2 * matvec(iM, matvec(H_, wf))
                dsig = signal(W, -T_[:, NAX] * dA, wf, phi0)
                dsig += signal(W, A, dwf, phi0)
                dsig += sig * 1j * dphi0[:, NAX]
                jac.append(dsig)

            if var == 'r2f':
                # signal gradient wrt r2f
                T_ = T**r2fpow
                dA = A * np.array([0, 1])
                y_ = vecvec(np.conj(A[..., 1] * W * T_), b)[:, NAX] * iM[..., 1]
                q_ = vecvec(AWb, y_)
                H_ = 1/2 * np.real((hermitian(AW) * T_) @ (W[..., NAX] * dA))
                H_ += hermitian(H_)
                s_ = vecvec(matvec(H_, z), z)

                dphi0 = np.imag((s_ - q_) / (p + ignore))
                dwf = np.real((-y_ - 1j * z * dphi0[:, NAX]) * P0)
                dwf += 2 * matvec(iM, matvec(H_, wf))
                dsig = signal(W, -T_[:, NAX] * dA, wf, phi0)
                dsig += signal(W, A, dwf, phi0)
                dsig += sig * 1j * dphi0[:, NAX]
                jac.append(dsig)

            if var == 'ndb':
                # signal gradient wrt ndb
                dfat = ndb_fat_model_grad(ndb, freqs, T)
                dA = a_matrix(T, 0, dfat, wchi=wchi, r2w=r2w, r2f=r2f)
                dAW = W[..., NAX] * dA
                # dM = np.real(hermitian(dAW) @ AW + hermitian(AW) @ dAW)
                dM = np.real(matmat(hermitian(dAW), AW) + matmat(hermitian(AW), dAW))
                dp = 2 * vecvec(matvec(hermitian(dAW), b), z)
                dp += - vecvec(matvec(dM, z), z)
                dphi0 = 0.5 * np.imag(dp / (p + ignore))
                dz = matvec(iM, matvec(hermitian(dAW), b))
                dz += - matvec(iM, matvec(dM, wf))
                dwf = np.real((dz - 1j * z * dphi0[:, NAX]) * P0)
                dsig = signal(W, dA, wf, phi0)
                dsig += signal(W, A, dwf, phi0)
                dsig += sig * 1j * dphi0[:, NAX]
                jac.append(dsig)
        
        jac = np.stack(jac, axis=-1)
        
        conf, ff = self.memory['conf'], self.memory['ff']
        for i, var in enumerate(self.variables):
            if var == 'b0' and self.opts['regularize_b0']:
                jac = np.pad(jac, [(0, 0), (0, 1), (0, 0)])
                jac[:, -1, i] = self.opts['mu_b0']
            if var == 'wchi' and self.opts['regularize_chi']:
                jac = np.pad(jac, [(0, 0), (0, 1), (0, 0)])
                jac[:, -1, i] = (1 - conf) * self.opts['mu_chi']
            if var == 'r2w' and self.opts['regularize_r2w']:
                jac = np.pad(jac, [(0, 0), (0, 1), (0, 0)])
                jac[:, -1, i] = ff * self.opts['mu_r2w']
                # jac[:, -1, i] = self.opts['mu_r2w']
            if var == 'r2f' and self.opts['regularize_r2f']:
                jac = np.pad(jac, [(0, 0), (0, 1), (0, 0)])
                jac[:, -1, i] = (1 - ff) * self.opts['mu_r2f']
                # jac[:, -1, i] = self.opts['mu_r2f']
            if var == 'ndb' and self.opts['regularize_ndb']:
                jac = np.pad(jac, [(0, 0), (0, 1), (0, 0)])
                jac[:, -1, i] = self.opts['mu_ndb']

        return jac

    def jac(self, x):
        raise NotImplementedError()
    
    def hess(self, x):
        """ jacobian/hessian operator """
        jac = self._jac(x)
        J = nlls.LinearOperator(jac)
        # H = nlls.LinearOperator(np.real(hermitian(jac) @ jac))
        H = nlls.LinearOperator(np.real(matmat(hermitian(jac), jac)))
        return J, H


# utils

def tovolume(arr, mask):
    arr = np.asarray(arr)
    vol = (0 * mask).astype(arr.dtype)
    vol[mask] = arr
    return vol

def to_filter_size(radius, spacing):
    radius = np.ones_like(spacing) * radius
    return 2 * np.round(radius / spacing).astype(int) + 1

def weighted_filter(arr, mask, *, conf=None, filter_size=3):
    """ box filter in mask"""
    mask = np.asanyarray(mask, dtype=bool)
    im = tovolume(arr, mask)
    conf = tovolume(conf, mask) / np.max(conf) if conf is not None else (mask * 1.0)
    weights = ndimage.uniform_filter(conf, size=filter_size, mode="nearest")
    imf = ndimage.uniform_filter(conf * im, size=filter_size, mode="nearest")
    valid = weights > 1e-10
    imf[valid] /= weights[valid]
    imf[~valid] = 0
    return imf[mask]

def guided_filter(arr, mask, guide=None, *, filter_size=3, sigma=1):
    """ guided filter in a mask"""
    im = tovolume(arr, mask)
    guide = im if guide is None else tovolume(guide, mask)
    guide = (guide - np.mean(guide[mask])) / np.std(guide[mask])

    filter_size = filter_size * np.ones(im.ndim)
    nfilter = np.prod(filter_size)

    weights = ndimage.uniform_filter(1.0 * mask, size=filter_size)
    valid = weights > 1e-5
    weights[valid & mask] = 1 / weights[valid & mask]
    weights[~valid | ~mask] = 0
    
    # mean and variance
    mean_gu = ndimage.uniform_filter(guide, size=filter_size) * weights
    var_gu = ndimage.uniform_filter((guide - mean_gu)**2, size=filter_size) * weights
    mean_im = ndimage.uniform_filter(im, size=filter_size) * weights
    mean_gu_im = ndimage.uniform_filter(guide * im, size=filter_size) * weights

    # estimate a, b
    a = (mean_gu_im - mean_gu * mean_im) / (var_gu + sigma * nfilter)
    b = mean_im - a * mean_gu
    a = ndimage.uniform_filter(a, size=filter_size) * weights
    b = ndimage.uniform_filter(b, size=filter_size) * weights

    # filtered image
    fim = a * guide + b
    return fim[mask]


#
# ls solvers

def solve_nnls(A, b):
    """non-negative least squares"""
    # M = np.real(hermitian(A) @ A)
    M = np.real(matmat(hermitian(A), A))
    Ab = np.real(matvec(hermitian(A), b))
    x = linsolve(M, Ab)
    
    # remove negative entries
    isneg = x < 0
    x1 = (Ab[..., 0] / M[..., 0, 0])[isneg[..., 1]]
    x2 = (Ab[..., 1] / M[..., 1, 1])[isneg[..., 0]] 
    x[isneg[..., 1], 0] = x1
    x[isneg[..., 0], 1] = x2
    x[isneg] = 0

    # residuals
    r = matvec(A, x) - b

    return x, r

def solve_pcls(A, b, *, p_filter=None):
    """non-negative least squares"""
    # M = np.real(hermitian(A) @ A)
    M = np.real(matmat(hermitian(A), A))
    Ab = matvec(hermitian(A), b)
    z = linsolve(M, Ab)
    p = vecvec(Ab, z)
    if p_filter is not None:
        p = p_filter(p)

    phi0 = 0.5 * np.angle(p)
    x = np.real(z * np.exp(-1j * phi0)[..., NAX])

    # residuals
    r = matvec(A, x) * np.exp(1j * phi0[:, NAX]) - b

    return x, phi0, r


#
# linalg

vecvec = pmath.vecvec
matvec = pmath.matvec
matmat = pmath.matmat
linsolve = pmath.linsolve
hermitian = pmath.hermitian


#    
# fat model

#
# Ren, Jimin, Ivan Dimitrov, A. Dean Sherry, et Craig R. Malloy. 
# « Composition of Adipose Tissue and Marrow Fat in Humans by 1H NMR at 7 Tesla * ». 
# Journal of Lipid Research 49, nᵒ 9 (2008): 2055‑62. https://doi.org/10.1194/jlr.D800010-JLR200.
MODEL_REN2008 = {
    'ampls': [125.8, 956.4, 109.1, 146.0, 100, 23.4, 63.9], # %
    'ppm': [0.9, 1.3, 1.59, 2.03, 2.25, 2.77, 5.31],  # ppm
}

# Yu, Huanzhou, Ann Shimakawa, Charles A. McKenzie, Ethan Brodsky, Jean H. Brittain, et Scott B. Reeder. 
# « Multiecho Water-Fat Separation and Simultaneous R 2* Estimation with Multifrequency Fat Spectrum Modeling ».
# Magnetic Resonance in Medicine 60, nᵒ 5 (2008): 1122‑34. https://doi.org/10.1002/mrm.21737.
MODEL_YU2008 = {
    'ampls': [0.087, 0.693, 0.128, 0.004, 0.039, 0.048], # %
    'ppm': [0.9, 1.3, 2.1, 2.76, 4.31, 5.3],  # ppm
}

#  Using a general model for measuring the intramuscular lipid spectrum:
#    Impact on the fat infiltration quantification in skeletal muscle
#    Noura Azzabou1,2, Harmen Reyngoudt1,2 , Pierre G. Carlier1,2
#    ISMRM 2017
MODEL_AZZABOU2017 = {
    'ampls': [0.0586, 0.0109, 0.0618, 0.1412, 0.66, 0.0673], 
    'ppm': [5.52, 3.01, 2.42, 2.22, 1.49, 1.08],  # ppm
    'T2': [46.3, 45.5, 40.1, 25.8, 81.4, 69.3], # ms
}

# Bydder, Mark, Olivier Girard, et Gavin Hamilton. 
# « Mapping the double bonds in triglycerides ». 
# Magnetic Resonance Imaging 29, nᵒ 8 (2011): 1041‑46. https://doi.org/10.1016/j.mri.2011.07.004.
MODEL_BYDDER2011 = {
    'ndb': 2.88,
    'ppm': [5.29, 5.19, 4.2, 2.75, 2.20, 2.02, 1.6, 1.3, 0.9],
    # 'T2': 80, # ms

}

# intra muscular fat model, in-house (+ added Glycerol 5.19)
MODEL_REYNGOUDT2024 = {
    'ndb': 2.88,
    'ppm': [5.51, 5.19, 4.01, 2.99, 2.43, 2.22, 1.75, 1.49, 1.08],
}

class FatModel:
    """ Fat model"""

    @classmethod
    def to_freqs(cls, field_strength, ppm):
        """ fat ppm to freqs in kHz """
        ppm = np.asarray(ppm)
        return  GAMMA_1H * field_strength * 1e-6 * (ppm - CS_WATER)
    
    @classmethod
    def to_ppm(cls, field_strength, freqs):
        """ fat freqs in kHz to ppm"""
        return freqs * 1e6 / GAMMA_1H / field_strength + CS_WATER
        
    def __init__(self, ampls, *, ppm=None, freqs=None, fs=3, T2=None):
        self.ampls = np.asarray(ampls) / np.sum(ampls)
        if ppm is not None:
            freqs = self.to_freqs(fs, ppm)
        self.freqs = np.asarray(freqs)
        self.r2 = 0 if T2 is None else 1 / np.asarray(T2)
            
    def __call__(self, times):
        """ fat signal """
        times = np.asarray(times)
        return sum([a * np.exp((2j * np.pi * f - self.r2) * times) for a, f in zip(self.ampls, self.freqs)])
        


class FatModelNDB(FatModel):
    """ Fat model with ndb-parameterized amplitudes

    Bydder, Mark, Olivier Girard, et Gavin Hamilton. 
    « Mapping the double bonds in triglycerides ». 
    Magnetic Resonance Imaging 29, nᵒ 8 (2011): 1041‑46. 
    https://doi.org/10.1016/j.mri.2011.07.004.

    """

    @property
    def ampls(self):
        ndb = self.ndb
        CL = 16.8 + 0.25 * ndb**2
        nmidb = 0.093 * ndb
        ampls = [2 * ndb, 1, 4, 2 * nmidb, 6, 4 * (ndb - nmidb), 6, 6 * (CL - 4) - 8 * ndb + 2 * nmidb, 9]
        tot = sum(ampls)
        return [a / tot for a in ampls]
        
    def __init__(self, ndb, ppm, *, fs=3, T2=None):
        ppm = np.array(ppm)
        freqs = self.to_freqs(fs, ppm)
        self.ndb = ndb
        self.freqs = freqs
        self.r2 = 0 if T2 is None else 1 / np.asarray(T2)


def ndb_fat_model(ndb, freqs, times):
    """ fat signal from ndb and frequencies """
    # compute amplitudes
    CL = 16.8 + 0.25 * ndb**2
    nmidb = 0.093 * ndb
    ampls = [2 * ndb, 1, 4, 2 * nmidb, 6, 4 * (ndb - nmidb), 6, 6 * (CL - 4) - 8 * ndb + 2 * nmidb, 9]
    tot = sum(ampls)
    ampls = [a / tot for a in ampls]
    # compute fat signal
    return sum([np.reshape(a, (-1, 1)) * np.exp(2j * np.pi * f * times) for a, f in zip(ampls, freqs)])

def ndb_fat_model_grad(ndb, freqs, times):
    """ gradient of fat signal """
    ndb = np.asarray(ndb)
    # amplitudes
    CL = 16.8 + 0.25 * ndb**2
    nmidb = 0.093 * ndb
    ampls = [2 * ndb, 1, 4, 2 * nmidb, 6, 4 * (ndb - nmidb), 6, 6 * (CL - 4) - 8 * ndb + 2 * nmidb, 9]
    tot = sum(ampls)
    # amplitudes derivatives
    dCL = 2 * 0.25 * ndb
    dnmidb = 0.093
    dampls = [2, 0, 0, 2 * dnmidb, 0, 4 * (1 - dnmidb), 0, 6 * dCL - 8 + 2 * dnmidb, 0]
    dtot = sum(dampls)
    dampls = [d/tot - dtot * a/tot**2 for a, d in zip(ampls, dampls)]
    # gradient of fat signal
    return sum([np.reshape(d, (-1, 1)) * np.exp(2j * np.pi * f * times).T for d, f in zip(dampls, freqs)])


#
# plotting

# def plot_optimization(obj, variables):
#     from matplotlib import pyplot as plt
#     plot_variables = ['FF', 'B0', 'R2', 'R2w', 'Xhi', 'Resids']

#     mask = obj.params['mask']
#     obs = obj.params['obs']

#     b0 = tovolume(variables['b0'], mask)
#     r2 = tovolume(variables['r2'], mask)
#     wchi = tovolume(variables['wchi'], mask)
#     r2w = tovolume(variables['r2w'], mask)
#     ndb = tovolume(variables['ndb'], mask)
#     phi0 = tovolume(variables['phi0'], mask)

#     wf = abs(variables['wf'])
#     ff = tovolume(wf[:,1] / np.clip(wf.sum(1), 1e-8, None), mask)

#     res = variables['rmsd'] / np.clip(np.linalg.norm(obs, axis=-1), 1e-8, None)
#     res = tovolume(res, mask)

#     cost = 0.5 * np.sum(variables['rmsd']**2)
    
#     slices = (slice(0, mask.shape[0]//2), slice(None), 21)
#     opts = {
#         'FF': {'vmin': 0, 'vmax': 1, 'cmap': 'gray', 'data': ff},
#         'B0': {'vmin': -0.5, 'vmax': 0.5, 'cmap': 'gray', 'data': b0},
#         'R2': {'vmin': 0, 'vmax': 0.1, 'cmap': 'gray', 'data': r2},
#         'R2w': {'vmin': 0.01, 'vmax': 0.05, 'cmap': 'gray', 'data': r2w},
#         'Xhi': {'vmin': -0.2, 'vmax': 0.2, 'cmap': 'gray', 'data': wchi},
#         'ndb': {'vmin': 2.4, 'vmax': 3.4, 'cmap': 'gray', 'data': ndb},
#         'Phi0': {'vmin': -3.15/2, 'vmax': 3.15/2, 'cmap': 'gray', 'data': phi0},
#         'Resids': {'vmin': 0, 'vmax': 1, 'cmap': 'gray', 'data': res},
#     }


#     if not plt.fignum_exists('optimization'):
#         fig, axes = plt.subplots(nrows=2, ncols=3, num='optimization', figsize=(10, 8))
#         for ax, title in zip(axes.flat, plot_variables):
#             plt.sca(ax)
#             plt.title(title)
#             h = plt.imshow(mask[*slices].T, interpolation='nearest', aspect='equal', **opts[title])
#             plt.axis('off')
#         plt.suptitle(f'setup')
#         plt.tight_layout()
#         plt.ion()
#         plt.show(block=False)
    
#     else:
#         fig = plt.figure('optimization')
    
#     fig.suptitle(f'iteration: {obj.iter}, cost={cost:.0f}')
#     axes = fig.get_axes()
#     for ax, name in zip(axes, plot_variables):
#         data = opts[name]['data']
#         ax.get_images()[0].set_data(data[*slices].T)
            
#     plt.draw()
#     plt.pause(1e-1)
    
    

# def plot_prediction(echo_times, volumes, res, roi, labels=None, real_imag=False):
#     """ plot predictions in ROI """
#     from matplotlib import pyplot as plt

#     labelset = list(set(np.unique(roi)) - {0})
#     labelnames = labels or {l: f'label {l}' for l in labelset}
#     nlabel = len(labelset)

#     echos = np.array(echo_times) 
#     times = echos
#     # times = np.linspace(np.min(echo_times), np.max(echo_times), 200)
#     fat = res['fat_model'](times)
    
#     # prediction
#     # A = np.stack([np.ones_like(fat_signal), fat_signal], axis=1)
#     wf = np.stack([res['wmap'], res['fmap']], axis=-1)
#     r2, phi0, b0map = res['r2'] * 1e-3, res['phi0'], res['b0map']
#     wchi, r2w = res.get('wchi', 0 * b0map), res.get('r2w', 0 * b0map)
#     # resids = res['resids']

#     nrows = int(nlabel**0.5 + 0.5)
#     ncols = - (-nlabel // nrows)
#     fig, axes = plt.subplots(nrows, ncols, layout='constrained', sharex=True, sharey=True)
#     for i in range(nlabel):
#         label = labelset[i]
#         labelname = labelnames[label]
#         mask = roi == label
#         sig = np.stack([vol[mask] for vol in volumes], axis=-1)
#         msig = np.mean(sig, axis=0)
#         # esig = np.std(sig, axis=0)
    
#         ff = wf[mask, 1] / (wf[mask, 0] + wf[mask, 1])
#         b0 = b0map[mask] - ff * wchi[mask]

#         # To fix
#         r2pow, r2wpow, r2fpow = params['r2pow'], params['r2wpow'], params['r2fpow']
#         W = w_matrix(b0, r2, T, r2pow=r2pow)
#         A = a_matrix(T, water, fat, wchi=wchi, r2w=r2w, r2f=r2f, r2wpow=r2wpow, r2fpow=r2fpow)
#         pred = signal(W, A, wf[mask], phi0[mask])
        
#         mpred = np.mean(pred, axis=0)
#         # epred = np.std(pred, axis=0)
#         # mresid = np.mean(resids[mask])

#         plt.sca(axes.flat[i])
#         if not real_imag:
#             plt.plot(echos, abs(msig), '+-', label='acquisition')
#             plt.plot(times, abs(mpred), '-', label='prediction')
#             plt.ylabel('magnitude (a.u.)')
#             plt.legend(loc='lower right')
#             plt.twinx()
#             plt.plot(echos, np.angle(msig), '.:')
#             plt.plot(times, np.angle(mpred), ':')
#             plt.ylabel('phase (rad)')
#             plt.ylim(-np.pi, np.pi)
#         else:
#             plt.plot(echos, np.real(msig), 'b+:', label='acquisition')
#             plt.plot(times, np.real(mpred), 'b-', label='prediction', alpha=0.5)
#             plt.legend(loc='lower right')
#             plt.ylabel('real / imag (a.u.)')
#             plt.plot(echos, np.imag(msig), 'g+:', label='acquisition')
#             plt.plot(times, np.imag(mpred), 'g-', label='prediction', alpha=0.5)
            

#         plt.xlabel('time (ms)')
#         plt.title(f'label: {labelname}')#, resids={100*mresid:.1f}%')
        

#     for i in range(nlabel, nrows*ncols):
#         plt.sca(axes.flat[i])
#         plt.axis('off')

#     return fig