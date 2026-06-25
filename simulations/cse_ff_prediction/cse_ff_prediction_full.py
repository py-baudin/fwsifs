"""
Simulate the effect on PDFF of incomplete or invalid signal models

usage:
python cse_ff_prediction_full.py

output: image file `fferr_full.png`

"""

import numpy as np
import pandas as pd
from fwsifs import nlls
# from scipy import optimize
# from scipy.sparse.linalg import LinearOperator



NAX = np.newaxis
LAX = (..., np.newaxis) # append axis
rstate = np.random.RandomState(0)

# constants
B0 = 3 # T
GAMMA_1H = 42.58e6 # Hz/T
CS_WATER = 4.7 
CHI_W = -9.05 # ppm
CHI_F = -8.44 # EMCL, Boesch, 1997
PPM2FREQ = 2 * np.pi * GAMMA_1H * B0 * 1e-6 # rad/s

# Bydder 2011
FAT_AMPL = [0.053, 0.009, 0.037, 0.005, 0.055, 0.095, 0.055, 0.61, 0.082]
FAT_PPM = [5.29, 5.19, 4.2, 2.75, 2.2, 2.02, 1.6, 1.3, 0.9]

# echo times
TE_IDEAL = [2.22, 5.42, 8.62, 11.82, 15.02, 18.22] # 6pt IDEAL

# 
# models 

def susceptibility_effects(angles, ff):
    """ wchi and R2s for given angle """
    delta_chi = CHI_F - CHI_W
    # delta frequency (ppm)
    delta_freq = - delta_chi * (1/3 - np.cos(angles  / 180 * np.pi)**2) * 1/2
    # delta frequency (rad/ms)
    wchi = delta_freq * PPM2FREQ * 1e-3 * np.ones_like(ff)

    # quadratic attenuation rate (1/ms^2)
    eta = 2 * np.pi * PPM2FREQ * delta_chi * np.sin(angles / 180 * np.pi)**2
    CGS2SI = 1e-3 * 2 * np.pi # CGS to SI (?)
    #r2p_lin = eta * CGS2SI
    r2p = 1/4 * eta**2 * CGS2SI
    # apply: frac -> (frac - frac^2 / 2) to imitate the ceiling effect at frac~0.5
    fceil = lambda ff: (ff - ff**2 / 2)
    r2pw = r2p * fceil(ff) * 1e-6
    r2pf = r2p * fceil(1 - ff) * 1e-6
    return wchi, r2pw, r2pf


def fat(t):
    """ fat signal for given t """
    t = np.asarray(t)[:, np.newaxis]
    ampls = np.asarray(FAT_AMPL)
    ampls /= np.sum(ampls)
    ppm = np.asarray(FAT_PPM) 
    freqs = (ppm - CS_WATER) * PPM2FREQ * 1e-3 # rad/ms
    return np.sum(ampls * np.exp(1j * freqs * t), axis=1)

def model(settings):
    """ signal model signal """
    t = settings['times']
    ff = settings['ff']
    angles = settings['angles']
    r2w, r2f = settings['r2w'], settings['r2f']
    wchi, r2pw, r2pf = susceptibility_effects(angles, ff)
    naxes = [slice(None)] + [NAX] * ff.ndim
    T = t[*naxes]

    # R2_power = 1
    # Tp = 10 
    # R2_power = 2
    Tp = T
    
    # signal model
    water_sig = (1 - ff) * np.exp((-r2w - r2pw * Tp) * T)
    fat_sig = ff * fat(t)[*naxes] * np.exp((-r2f  - r2pf * Tp + 1j * wchi) * T)
    signal = water_sig + fat_sig # B0 = 0 
    
    # set total frequency shift to 0 (ie. add corresponding shift of water)
    signal *= np.exp(-1j * wchi * T * ff)

    # replicate signal
    signal = signal * np.ones_like(nexp)
    
    # add noise
    noise = rstate.normal(size=signal.shape) + 1j * rstate.normal(size=signal.shape)
    noise /= np.linalg.norm(noise, axis=0, keepdims=True)
    norm = np.linalg.norm(signal, axis=0, keepdims=True)
    # noise[...,0] = 0 # first point is without noise
    signal = signal + noise * norm * 1/snr

    params = dict(ff=ff, wchi=wchi, r2pw=r2pw, r2pf=r2pf)
    return signal, params


#
# problem

class Params:
    def __init__(self, variables, init):
        self.variables = variables
        self._values = init
    @property
    def values(self):
        return np.stack([self._values[var] for var in self.variables], -1)
    def update(self, args):
        for i, var in enumerate(self.variables):
            self._values[var] = args[:, i]
    def __getitem__(self, name):
        return self._values[name]
    def __setitem__(self, name, value):
        self._values[name] = value
                        
from fwsifs.pmath import matmat, matvec, vecvec, linsolve, hermitian

def solve_npt_pdff(name, settings, variables, init=None):
    # sample signal
    times = settings['times']
    sig, gt = model(settings)
    
    # solve F/W separation (no unwrapping since B0=0)
    n = sig[0].size
    shape = sig.shape[1:]

    T = np.array(times)
    b = np.reshape(sig, (len(times), -1)).T

    # init parameters
    zeros = np.zeros(n)
    init = dict(b0=zeros, wchi=zeros, r2p=zeros, r2pw=zeros, r2pf=zeros, **(init or {}))
    params = Params(variables, init)

    def pcls(params):
        # PCLS 
        b0 = params['b0']
        wchi = params['wchi']
        r2w, r2f = params['r2w'], params['r2f']
        r2p, r2pw, r2pf = params['r2p'], params['r2pw'], params['r2pf']
        r2pow = params['r2pow']

        # clip r2pw, r2pf
        ubound = 0.2 * np.mean(times)**(1 - r2pow)
        r2pw, r2pf = np.clip(r2pw, 0, ubound), np.clip(r2pf, 0, ubound)
        params['r2pw'], params['r2pf'] = r2pw, r2pf

        W = np.exp((1j * b0 - r2p)[LAX] * T)  
        Aw = np.exp(- r2w * T - r2pw[LAX] * T**r2pow)
        Af = np.exp((1j * wchi - r2f)[LAX] * T - r2pf[LAX] * T **r2pow) * fat(T)
        A = np.stack([Aw, Af], axis=-1)

        WA = W[LAX] * A
        iM = np.linalg.inv(matmat(hermitian(WA), WA).real)
        AWb = matvec(hermitian(WA), b)
        z = matvec(iM, AWb) 
        p = vecvec(AWb, z)
        phi0 = 0.5 * np.angle(p)
        x = np.real(z * np.exp(-1j * phi0[LAX]))
        ff = abs(x[:, 1]) / (abs(x[:, 0]) + abs(x[:, 1]))
        return dict(A=A, W=W, WA=WA, AWb=AWb, iM=iM, z=z, p=p, x=x, phi0=phi0, ff=ff)
    
    class Objective(nlls.Objective):

        def init(self):
            return params.values

        def pred(self, args):
            params.update(args)
            temp = pcls(params)
            W, A, x, phi0 = temp['W'], temp['A'], temp['x'], temp['phi0'] 
            return matvec(W[LAX] * A, x * np.exp(1j * phi0[LAX]))

        def fun(self, args):
            res = self.pred(args) - b
            return res
        
        def jac(self, args):
            raise NotImplementedError()
        
        def hess(self, args):
            params.update(args)
            temp = pcls(params)
            W, A, WA, AWb, iM = temp['W'], temp['A'], temp['WA'], temp['AWb'], temp['iM']
            p, z, x, phi0 = temp['p'], temp['z'], temp['x'], temp['phi0']
            r2pow = params['r2pow']

            pred = matvec(W[LAX] * A, x * np.exp(1j * phi0[LAX]))
            gradients = []
            for var in variables:
                if var == 'b0':
                    AWTb = matvec(hermitian(WA) * T, b)
                    y = matvec(iM, AWTb) 
                    q = vecvec(AWb, y)
                    dphi0 = - np.real(q / p)
                    dx = np.imag((y + z * dphi0[LAX]) * np.exp(-1j * phi0[LAX]))
                    grad = 1j * matvec((W * T)[LAX] * A, x * np.exp(1j * phi0[LAX]))
                    grad += matvec(W[LAX] * A, dx * np.exp(1j * phi0[LAX]))
                    grad += pred * 1j * dphi0[LAX]
                    gradients.append(grad)

                if var == 'r2p':
                    AWTb = matvec(hermitian(WA) * T, b)
                    y = matvec(iM, AWTb) 
                    q = vecvec(AWb, y)
                    H = matmat(hermitian(WA) * T, WA).real
                    s = vecvec(z, matvec(H, z))
                    dphi0 = np.imag((s - q) / p)
                    dx = - np.real((y + 1j * z * dphi0[LAX]) * np.exp(-1j * phi0[LAX]))
                    dx += 2 * matvec(iM, matvec(H, x))
                    grad = - matvec((W * T)[LAX] * A, x * np.exp(1j * phi0[LAX]))
                    grad += matvec(W[LAX] * A, dx * np.exp(1j * phi0[LAX]))
                    grad += pred * 1j * dphi0[LAX]
                    gradients.append(grad)

                if var == 'wchi':
                    Af = A * np.array([0, 1])
                    y_ = 1j * vecvec(np.conj(A[..., 1] * W * T), b)[LAX] * iM[..., 1]
                    q_ = vecvec(AWb, y_)
                    H_ = - 1/2 * matmat(hermitian(WA) * T, W[LAX] * Af).imag
                    H_ += 1/2 * matmat(hermitian(W[LAX] * Af) * T, WA).imag
                    s_ = vecvec(matvec(H_, z), z)
                    dphi0 = - np.imag((s_ + q_) / p)
                    dx = - np.real((y_ + 1j * z * dphi0[LAX]) * np.exp(-1j * phi0[LAX]))
                    dx -= 2 * matvec(iM, matvec(H_, x))
                    grad = 1j * matvec((W * T)[LAX] * Af, x * np.exp(1j * phi0[LAX]))
                    grad += matvec(W[LAX] * A, dx * np.exp(1j * phi0[LAX]))
                    grad += pred * 1j * dphi0[LAX]
                    gradients.append(grad)

                if var == 'r2pw':
                    T_ = T**r2pow
                    Aw = A * np.array([1, 0])
                    y_ = vecvec(np.conj(A[..., 0] * W * T_), b)[LAX] * iM[..., 0]
                    q_ = vecvec(AWb, y_)
                    H_ = 1/2 * matmat(hermitian(WA) * T_, W[LAX] * Aw).real
                    H_ += H_.mT
                    s_ = vecvec(matvec(H_, z), z)
                    dphi0 = np.imag((s_ - q_) / p)
                    dx = - np.real((y_ + 1j * z * dphi0[LAX]) * np.exp(-1j * phi0[LAX]))
                    dx += 2 * matvec(iM, matvec(H_, x))
                    grad = - matvec((W * T_)[LAX] * Aw, x * np.exp(1j * phi0[LAX]))
                    grad += matvec(W[LAX] * A, dx * np.exp(1j * phi0[LAX]))
                    grad += pred * 1j * dphi0[LAX]
                    gradients.append(grad)

                if var == 'r2pf':
                    T_ = T**r2pow
                    Af = A * np.array([0, 1])
                    y_ = vecvec(np.conj(A[..., 1] * W * T_), b)[LAX] * iM[..., 1]
                    q_ = vecvec(AWb, y_)
                    H_ = 1/2 * matmat(hermitian(WA) * T_, W[LAX] * Af).real
                    H_ += H_.mT
                    s_ = vecvec(matvec(H_, z), z)
                    dphi0 = np.imag((s_ - q_) / p)
                    dx = - np.real((y_ + 1j * z * dphi0[LAX]) * np.exp(-1j * phi0[LAX]))
                    dx += 2 * matvec(iM, matvec(H_, x))
                    grad = - matvec((W * T_)[LAX] * Af, x * np.exp(1j * phi0[LAX]))
                    grad += matvec(W[LAX] * A, dx * np.exp(1j * phi0[LAX]))
                    grad += pred * 1j * dphi0[LAX]
                    gradients.append(grad)

            gradients = np.stack(gradients, axis=-1)

            jac = nlls.LinearOperator(gradients)
            hess = nlls.LinearOperator((matmat(hermitian(gradients), gradients)).real)

            return jac, hess

    # solve
    print('optimize PDFF')
    objective = Objective()
    res = nlls.nlls(objective, disp=True, maxiter=30)
    
    # store results
    params.update(res.x)
    res_pcls = pcls(params)
    ff = res_pcls['ff'].reshape(shape)

    ones = np.ones_like(ff, dtype=int)
    FF = [f'{v:.0%}' for v in (ones * settings['ff']).ravel()]
    ANGLES = [f'{a:.0f}' for a in (ones * settings['angles']).ravel()]
    NEXP = [f'{n}' for n in (ones * settings['nexp']).ravel()]
    SNR = settings['snr']
    return pd.DataFrame({
        'NAME': name,
        'FF': FF,
        'FF_ERR': (ff - settings['ff']).ravel() * 100, 
        'ANGLE': ANGLES,
        'SNR': SNR,
        'NEXP': NEXP,
    })




# variables
times = np.array(TE_IDEAL)
magic_angle = np.acos((1/3)**.5) / np.pi * 180
angles = np.r_[magic_angle, np.linspace(0, 90, 7)]
 
ff = np.array([1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 - 1e-2])
snr = 20
nexp = np.arange(50)
r2w, r2f = 1/25, 1/80 # 1/ms

ff = ff[:, NAX, NAX]
angles = angles[NAX, :, NAX]
nexp = nexp[NAX, NAX, :]

settings = dict(times=times, ff=ff, angles=angles, snr=snr, nexp=nexp, r2w=r2w, r2f=r2f)

# List of tests
# cf. description in comments
tests = dict(
    # fit only R2'
    r2p={'variables': ['r2p'], 'title': r'$R^{\prime}_2$'},

    # fit B0 and R2'
    b0_r2p={'variables': ['b0', 'r2p'], 'title': r'${\Delta\omega}_0, R^{\prime}_2$'},

    # fit B0, w_chi and R2'
    b0_r2p_wchi={
        'variables': ['b0', 'r2p', 'wchi'], 
        'title': r'${\Delta\omega}_0, {\Delta\omega}_\chi, R^{\prime}_2$'},

    # fit B0, w_chi and R2'w / R2'f (dual relaxations, linear decay function)
    b0_wchi_dualr2={
        'variables': ['b0', 'wchi', 'r2pw', 'r2pf'], 
        'title': r'${\Delta\omega}_0, {\Delta\omega}_\chi, \eta_w(t)=R^{\prime}_{2w}t, \eta_f(t)=R^{\prime}_{2f}t$'},

    # fit B0, w_chi and R2'w / R2'f (dual relaxations, quadratic decay function)
    # use correct molecular R2 values
    b0_wchi_dualr2_pow2={
        'variables': ['b0', 'wchi', 'r2pw', 'r2pf'], 'init': {'r2pow': 2},
        'title': r'${\Delta\omega}_0, {\Delta\omega}_\chi, \eta_w(t)=R^{\prime}_{2w}t^2, \eta_f(t)=R^{\prime}_{2f}t^2$'
    },

    # fit B0, w_chi and R2'w / R2'f (dual relaxations, quadratic decay function)
    # use wrong (low) molecular R2f
    b0_wchi_dualr2_pow2_lowR2f={
        'variables': ['b0', 'wchi', 'r2pw', 'r2pf'], 'init': {'r2pow': 2, 'r2f': r2f * 2/3 },
        'title': r'${\Delta\omega}_0, {\Delta\omega}_\chi, \eta_w(t)=R^{\prime}_{2w}t^2, \eta_f(t)=R^{\prime}_{2f}t^2$'
                 '\n' 
                 r'$R_{2f}=2/3\overline{R}_{2f}$'
    },

    # fit B0, w_chi and R2'w / R2'f (dual relaxations, quadratic decay function)
    # use wrong (high) molecular R2f
    b0_wchi_dualr2_pow2_highR2f={
        'variables': ['b0', 'wchi', 'r2pw', 'r2pf'], 'init': {'r2pow': 2, 'r2f': r2f * 3/2 },
        'title': r'${\Delta\omega}_0, {\Delta\omega}_\chi, \eta_w(t)=R^{\prime}_{2w}t^2, \eta_f(t)=R^{\prime}_{2f}t^2$'
                 '\n' 
                 r'$R_{2f}=3/2\overline{R}_{2f}$'
    },

    

    # fit B0, w_chi and R2'w / R2'f (dual relaxations, quadratic decay function)
    # use wrong (high) molecular R2w
    b0_wchi_dualr2_pow2_lowR2w={
        'variables': ['b0', 'wchi', 'r2pw', 'r2pf'], 'init': {'r2pow': 2, 'r2w': r2w * 2/3 },
        'title': r'${\Delta\omega}_0, {\Delta\omega}_\chi, \eta_w(t)=R^{\prime}_{2w}t^2, \eta_f(t)=R^{\prime}_{2f}t^2$'
                 '\n' 
                 r'$R_{2w}=2/3\overline{R}_{2w}$'
    },

     # fit B0, w_chi and R2'w / R2'f (dual relaxations, quadratic decay function)
    # use wrong (low) molecular R2w
    b0_wchi_dualr2_pow2_highR2w={
        'variables': ['b0', 'wchi', 'r2pw', 'r2pf'], 'init': {'r2pow': 2, 'r2w': r2w * 3/2 },
        'title': r'${\Delta\omega}_0, {\Delta\omega}_\chi, \eta_w(t)=R^{\prime}_{2w}t^2, \eta_f(t)=R^{\prime}_{2f}t^2$'
                 '\n' 
                 r'$R_{2w}=3/2\overline{R}_{2w}$'
    },



)

results = []
for name, test in tests.items():
    init = {**dict(r2w=r2w, r2f=r2f, r2pow=1), **test.get('init', {})}
    df = solve_npt_pdff(name, settings, test['variables'], init=init)
    df['TITLE'] = test['title']
    results.append(df)
df = pd.concat(results, axis=0)

#
# plotting
import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')
sns.set_theme()
sns.set_style("whitegrid", rc={"grid.color": "#eeeeee"})
sns.set_context("paper")#, rc={"lines.linewidth": 0.7})

names = df.NAME.unique()
ncols = int(len(names)**0.5 + 0.5)
nrows = -(-len(names) // ncols)
figsize = (8, 3 * nrows)

fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, 
    squeeze=False, sharex=True, sharey=True,
    num='fferr-full', 
    figsize=figsize,
    layout='compressed',
)

ticks = np.arange(6)*2 / 10
erropts = dict(ha='right', fontsize='x-small', color='black', bbox=dict(facecolor='w', alpha=0.3), alpha=0.4)
errloc = len(ff) * 0.95


for i, name in enumerate(names):
    islegend = (i == ncols - 1)
    gp = df[df.NAME==name]
    bias, conf = gp['FF_ERR'].mean(), 1.96 * gp['FF_ERR'].std()

    plt.sca(axes.flat[i])
    ax = sns.lineplot(gp, hue='ANGLE', style='ANGLE', y='FF_ERR', x='FF', legend=islegend)

    # bias/LOA
    plt.axhline(bias, color='silver', zorder=1)
    plt.text(errloc, bias, f'MEAN\n{bias:0.2f}', va='center', **erropts)
    plt.axhline(bias + conf, color='silver', linestyle=':', zorder=1)
    plt.text(errloc, bias + conf, f'+SD1.96\n{bias + conf:0.2f}', va='bottom', **erropts)
    plt.axhline(bias - conf, color='silver', linestyle=':', zorder=1)
    plt.text(errloc, bias - conf, f'-SD1.96\n{bias - conf:0.2f}', va='top', **erropts)

    sns.despine()
    plt.xlabel('$PDFF$ [%]')
    plt.ylabel('$PDFF$ error [%]')
    plt.xticks(np.linspace(0, len(ff) - 1, len(ticks)), [f'{100*t:.0f}' for t in ticks])
    plt.title(gp.TITLE.iloc[0], loc='left', fontsize='small')
    plt.text(-0.1, 1.05, f'{i + 1}', transform=plt.gca().transAxes, weight='bold')

    if islegend:
        sns.move_legend(ax, bbox_to_anchor=(1,1), loc='upper left', title='angle to $B_0$')

for i in range(len(names), axes.size):
    plt.sca(axes.flat[i])
    plt.axis('off')

fig.get_layout_engine().set(hspace=0.05, wspace=0)
fig.suptitle('$PDFF$ errors for various models (SNR=20)')


#
#  plot wchi and r2p variables

angles = np.linspace(0, 90, 6)
ffracs = np.linspace(0, 1, 11)
df = pd.DataFrame([dict(FF=ff, ANGLE=a) for a in angles for ff in ffracs])
df[['WCHI', 'R2PW', 'R2PF']] = pd.concat(susceptibility_effects(df['ANGLE'], df['FF']), axis=1)
df['WCHI_PPM'] = df.WCHI / PPM2FREQ * 1e3
df[['ETA_W', 'ETA_F']] = df[['R2PW', 'R2PF']] * 10**2

fig, axes = plt.subplots(nrows=3, num='fferr-full-susc-effects', layout='constrained', sharex=True)
plt.sca(axes[0])
ax = sns.lineplot(data=df, x='ANGLE', y='WCHI_PPM', hue='FF', legend=True)
handles = ax.legend_.legend_handles
ax.get_legend().remove()
plt.ylabel(r'${\Delta\omega}_\chi [ppm]$')
plt.title('Frequency shift between fat and water')
plt.sca(axes[1])
sns.lineplot(data=df, x='ANGLE', y='ETA_W', hue='FF', legend=False)
plt.ylabel(r'$\eta_w(t=10ms)$')
plt.title('Water decay at t=10ms')
plt.sca(axes[2])
sns.lineplot(data=df, x='ANGLE', y='ETA_F', hue='FF', legend=False)
plt.ylabel(r'$\eta_f(t=10ms)$')
plt.xlabel(r'Angle to $B_0 [\degree]$ ')
plt.title('Fat decay at t=10ms')
fig.legend(loc='outside right upper', title='PDFF', handles=handles)
fig.suptitle('Susceptibility-induced effects')


for fignum in plt.get_fignums():
    fig = plt.figure(fignum)
    filename = fig.get_label()
    print(f"writing figure: {filename}")
    # plt.savefig(filename + '.svg')
    plt.savefig(filename + '.png', dpi=600)
