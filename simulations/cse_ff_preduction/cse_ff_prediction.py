"""
Simulate the effect on PDFF and R2 estimation error of unaccounted frequency shift between fat and water

usage:
python cse_ff_prediction.py

output: image file `fferr.png`

"""

import numpy as np
import pandas as pd
from scipy import optimize



NAX = np.newaxis
rstate = np.random.RandomState(0)

# variables
chi = np.array([-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2]) # ppm
r2 = 40 * np.ones(len(chi))
ff = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
snr = np.array([20])
nexp = np.arange(50)


VARS = [ff[:, NAX, NAX, NAX], chi[NAX, :, NAX, NAX], r2[NAX, :, NAX, NAX], snr[NAX, NAX, :, NAX], nexp[NAX, NAX, NAX, :]]

# constants
B0 = 3 # T
GAMMA_1H = 42.58e6 # Hz/T
CS_WATER = 4.7 

# # Fat model 
# # Ren, Jimin, Ivan Dimitrov, A. Dean Sherry, et Craig R. Malloy. 
# # « Composition of Adipose Tissue and Marrow Fat in Humans by 1H NMR at 7 Tesla * ». 
# # Journal of Lipid Research 49, nᵒ 9 (2008): 2055‑62. https://doi.org/10.1194/jlr.D800010-JLR200.
# FAT_AMPL = [125.8, 956.4, 109.1, 146.0, 100, 23.4, 63.9], # %
# FAT_PPM = [0.9, 1.3, 1.59, 2.03, 2.25, 2.77, 5.31],  # ppm

# Bydder 2011
FAT_AMPL = [0.053, 0.009, 0.037, 0.005, 0.055, 0.095, 0.055, 0.61, 0.082]
FAT_PPM = [5.29, 5.19, 4.2, 2.75, 2.2, 2.02, 1.6, 1.3, 0.9]

# echo times
TE_DIX1 = [2.75, 3.95, 5.15] # 3pt Dixon
TE_IDEAL = [2.22, 5.42, 8.62, 11.82, 15.02, 18.22] # 6pt IDEAL

def fat(t):
    """ fat signal for given t """
    t = np.asarray(t)[:, np.newaxis]
    ampls = np.asarray(FAT_AMPL)
    ampls /= np.sum(ampls)
    ppm = np.asarray(FAT_PPM) 
    freqs = (ppm - CS_WATER) * 1e-6 * GAMMA_1H * B0 * 1e-3
    return np.sum(ampls * np.exp(2j * np.pi * freqs * t), axis=1)

def model(t, ff, chi, r2, snr, nexp):
    """ signal model signal """
    t = np.asarray(t)
    ff = np.asarray(ff)
    chi = np.asarray(chi)
    r2 = np.asarray(r2)
    nexp = np.asarray(nexp)
    naxes = [slice(None)] + [NAX] * ff.ndim

    # frequency shift
    om = chi * 1e-6 * B0 * GAMMA_1H * 2 * np.pi * 1e-3

    # signal model
    signal = np.exp(- r2 * t[*naxes] * 1e-3) * ((1 - ff) + ff * fat(t)[*naxes] * np.exp(1j * om * t[*naxes])) * np.ones_like(nexp)
    
    # set total frequency shift to 0 (ie. add corresponding shift of water)
    signal *= np.exp(-1j * om * t[*naxes] * ff)

    if snr is None:
        return signal
    
    # noise
    snr = np.asarray(snr)
    noise = rstate.normal(size=signal.shape) + 1j * rstate.normal(size=signal.shape)
    noise /= np.linalg.norm(noise, axis=0, keepdims=True)
    norm = np.linalg.norm(signal, axis=0, keepdims=True)
    return signal + noise * norm * 1/snr


def solve_3pt_dixon(times):
    """ Estimation error for 3pt dixon
    Glover, Gary H. 
    “Multipoint Dixon Technique for Water and Fat Proton and Susceptibility Imaging.” 
    Journal of Magnetic Resonance Imaging 1, no. 5 (1991): 521–30. 
    https://doi.org/10.1002/jmri.1880010504.

    """
    # sample signal
    ff, chi, r2, snr, nexp = VARS
    sig = model(times, ff, chi, r2, snr, nexp)

    # solve F/W separation (no unwrapping since B0=0)
    A = np.sqrt(abs(sig[2]) / abs(sig[0]))
    sign = np.cos(np.angle(sig[1] * np.conj(sig[0])))
    w = abs(sig[0]) + sign * abs(sig[1]) / A
    f = abs(sig[0]) - sign * abs(sig[1]) / A
    ff_ = f / (w + f)
    r2_ = - np.log(A) / np.diff(times[:2]) * 1e3

    # tore results
    ones = np.ones_like(ff_, dtype=int)
    FF = [f'{v:.0%}' for v in (ones*ff).ravel()]
    CHI = [f'{v:+.2f}' for v in (ones*chi).ravel()]
    R2 = [f'{v:.0f}' for v in (ones*r2).ravel()]
    CHI_R2 = [f'{chi}|{r2}' for chi, r2 in zip(CHI, R2)]
    return pd.DataFrame({
        'FF': FF,
        'CHI': CHI,
        'R2': R2,
        'CHI_R2': CHI_R2,
        'SNR': (ones*snr).ravel(),
        'NEXP': (ones*nexp).ravel(),
        'FF_ERR': (ff_ - ff).ravel() * 100, 
        'R2_ERR': (r2_ - r2).ravel(), 
    })


def solve_npt_pdff(times):
    """
    """
    # sample signal
    ff, chi, r2, snr, nexp = VARS
    sig = model(times, ff, chi, r2, snr, nexp)
    
    # solve F/W separation (no unwrapping since B0=0)
    shape = sig.shape[1:]
    T = np.expand_dims(np.array(times), list(range(1, sig.ndim)))
    A = np.stack([np.ones_like(times), fat(times)], axis=1)

    def pcls(b0, r2):
        # PCLS 
        W = np.exp((1j * b0 - r2) * T)
        iM = np.linalg.inv(np.einsum('ex,e...,ey->...xy', A.conj(), abs(W)**2 , A).real)
        AWb = np.einsum('ex,e...->x...', A.conj(), W.conj() * sig)
        z = np.einsum('...xy,y...->x...', iM, AWb)
        p = np.einsum('x...,x...', AWb, z)
        phi0 = 0.5 * np.angle(p)
        x = np.real(z * np.exp(-1j * phi0[NAX]))
        ff = abs(x[1]) / (abs(x[0]) + abs(x[1]))
        return ff, dict(A=A, W=W, iM=iM, z=z, p=p, x=x, phi0=phi0)
    
    def objective(args):
        # return residuals
        r2 = args.reshape(shape)
        _, dct = pcls(0, r2)
        W, x, phi0 = dct['W'], dct['x'], dct['phi0']
        
        pred = W * np.einsum('xy,y...->x...', A, x) * np.exp(1j * phi0)
        res = (pred - sig).ravel()
        return np.r_[res.real, res.imag] 
    
    def jac(args):
        # jacobian matrix
        r2 = args.reshape(shape)
        
        ff, dct = pcls(0, r2)
        W, iM, p, z = dct['W'], dct['iM'], dct['p'], dct['z']
        x, phi0 = dct['x'], dct['phi0']

        pred = W * np.einsum('xy,y...->x...', A, x) * np.exp(1j * phi0)

        AWb = np.einsum('ex,e...->x...', A.conj(), W.conj() * sig)
        AWTb = np.einsum('ex,e...->x...', A.conj(), T * W.conj() * sig)
        y = np.einsum('...xy,y...->x...', iM, AWTb) 
        q  = np.einsum('x...,x...', AWb, y)
        H = np.einsum('ex,e...,ey->...xy', A.conj(), T * abs(W)**2 , A).real
        s = np.einsum('x...,...xy,y...->...', z, H, z)
        dphi0 = np.imag((s - q) / p)
        dx = np.real((-y - 1j * z * dphi0) * np.exp(-1j * phi0))
        dx += 2 * np.einsum('...xy,...yz,z...->x...', iM, H, x)

        # gradient of prediction
        grad = -W * T * np.einsum('xy,y...->x...', A, x) * np.exp(1j * phi0)
        grad += W * np.einsum('xy,y...->x...', A, dx) * np.exp(1j * phi0)
        grad += pred * 1j * dphi0

        jac = np.zeros((len(times), r2.size, r2.size), dtype=pred.dtype)
        jac[:, np.arange(r2.size), np.arange(r2.size)] = grad.reshape(-1, r2.size)
        jac = jac.reshape(-1, r2.size)
        return np.r_[jac.real, jac.imag] 

    # solve
    n = sig[0].size
    r2_init = 40 * 1e-3 * np.ones(n)
    init = r2_init
    print('optimize PDFF')
    res = optimize.least_squares(objective, init, jac=jac, verbose=2)
    r2_ = res.x.reshape(shape)
    ff_, _ = pcls(0, r2_)

    # store results
    ones = np.ones_like(ff_, dtype=int)
    FF = [f'{v:.0%}' for v in (ones*ff).ravel()]
    CHI = [f'{v:+.2f}' for v in (ones*chi).ravel()]
    R2 = [f'{v:.0f}' for v in (ones*r2).ravel()]
    CHI_R2 = [f'{chi}|{r2}' for chi, r2 in zip(CHI, R2)]
    return pd.DataFrame({
        'FF': FF,
        'CHI': CHI,
        'R2': R2,
        'CHI_R2': CHI_R2,
        'SNR': (ones*snr).ravel(),
        'NEXP': (ones*nexp).ravel(),
        'FF_ERR': (ff_ - ff).ravel() * 100, 
        'R2_ERR': (r2_*1e3 - r2).ravel(), 
    })


df_dix3p = solve_3pt_dixon(TE_DIX1)
df_nlls6 = solve_npt_pdff(TE_IDEAL)

keys = ['FF', 'CHI', 'SNR']
with open('medians.txt', 'w') as fp:
    print('Medians for 3pt-Dixon:', file=fp)
    print(df_dix3p.groupby(keys, sort=False)['FF_ERR'].median(), file=fp)

    print('Medians for 6pt-IDEAL:', file=fp)
    print(df_nlls6.groupby(keys, sort=False)['FF_ERR'].median(), file=fp)


#
# plotting 
FIGSIZE = (6.3, 2.5)


import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')
sns.set_theme()
sns.set_style("whitegrid", rc={"grid.color": "#eeeeee"})
sns.set_context("paper", font_scale=0.8, rc={"lines.linewidth": 0.7})

chi_order = list(df_dix3p.CHI.unique())
dashes = [(5, 5)] * len(chi)
dashes[chi_order.index('+0.00')] = ''

fig = plt.figure(num='fferr', figsize=FIGSIZE, layout='compressed')

ax1 = plt.subplot(121)
sns.lineplot(df_dix3p, hue='CHI', style='CHI', style_order=chi_order, dashes=dashes, y='FF_ERR', x='FF', legend=False)
sns.despine(offset=10)
plt.xlabel('$FF$ [%]')
plt.ylabel('$FF$ error [%]')
plt.title('3-point Dixon')

ticks = np.arange(6)*2 / 10
plt.xticks(np.linspace(0, len(ff) - 1, len(ticks)), [f'{100*t:.0f}' for t in ticks])

ax2 = plt.subplot(122)
ax2.sharex(ax1)
ax2.sharey(ax1)
sns.lineplot(df_nlls6, hue='CHI', style='CHI', style_order=chi_order, dashes=dashes, y='FF_ERR', x='FF', legend='full')
sns.despine(offset=10)
plt.title('6-point NLLS')
sns.move_legend(ax2, 'upper left', bbox_to_anchor=(1, 1), fontsize='small', title=r'$\omega_\chi$ [ppm]')
plt.xlabel('$FF$ [%]')
plt.ylabel('')

print('3pt dixon, chi=-0.1ppm (median):')
print(df_dix3p[df_dix3p.CHI == '-0.10'].groupby('FF')['FF_ERR'].median())

print('6pt nlls, chi=+0.2ppm (median):')
print(df_nlls6[df_nlls6.CHI == '+0.20'].groupby('FF')['FF_ERR'].median())


for fignum in plt.get_fignums():
    fig = plt.figure(fignum)
    filename = fig.get_label()
    print(f"writing figure: {filename}")
    plt.savefig(filename + '.svg')
    plt.savefig(filename + '.png', dpi=600)
