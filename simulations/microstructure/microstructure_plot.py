"""
create figure from existing simulation results
usage:
python microstructure_plot.py objRCYLINDERS_3_nff5_img1001_bin_nexp20.pickle

output:
image file:  `objRCYLINDERS_3_nff5_img1001_bin_nexp20.png`

"""

import sys
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

SHOW = False

# constants
FIGSIZE = (6.3, 4)
PLOT_OBJECT = (0, 0.5)

# chi (ppm)
CHI_W = -9.05
# CHI_F = -7.79 #  Fat, Hoptkins 1997
CHI_F = -8.44 # EMCL, Boesch, 1997

B0 = 2.89
GAMMA_1H = 42.58e6 # Hz/T
ppmfreq =  2 * np.pi * GAMMA_1H * B0 * 1e-6


# load data
exp_file = sys.argv[1]
exp_name = exp_file[:-len('.pickle')]
print(f'Loading data for experiment: {exp_name}')

with open(exp_file, 'rb') as fp:
    data = pickle.load(fp)

info = data['info']
stats = data['stats']
fids = data['fids']
spectra = data['spectra']
objects = data['objects']

masks = {(obj['nexp'], obj['fraction']): obj['mask'] for obj in objects}
cube = np.ones([max(info['SHAPE'])]*3, dtype=bool)
mask = masks[PLOT_OBJECT] * cube


NAMES = {
    'RCYLINDERS': 'Rand. cylinders',
    'AGGCYLINDERS': 'Aggr. cylinders',
    'PERLIN2D': '2D Perlin noise',
}
name = NAMES.get(info['OBJECT'], info['OBJECT'])

fraction_pc = [f'{f:.0%}' for f in fids.fraction.unique()]
if not 'PERLIN' in info['OBJECT']:
    fids = fids[fids.fraction <= 0.5]
    stats = stats[stats.fraction <= 0.5]


#
# plot
print('Plotting')
sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=0.8, rc={"lines.linewidth": 1})
plt.close('all')

fig, axes = plt.subplots(nrows=2, ncols=3, num='microstructure', layout='constrained', figsize=FIGSIZE)

# plot object
plt.sca(axes[0,0])

plt.imshow(mask[:,:,mask.shape[2]//2], cmap='gray')
plt.xticks([])
plt.yticks([])

plt.title(f'{name} (FF={PLOT_OBJECT[1]:.0%})')

# plot freq shift vs angle
angles = info['ANGLES']
shift_cylinder = 1/2 * (np.cos(angles * np.pi / 180)**2 - 1/3) * (CHI_F - CHI_W)

ax = axes[1, 0]
plt.sca(ax)
sns.lineplot(data=stats, x='angle', y='shift', hue='fraction', palette='flare_r', legend=False)
sns.lineplot(x=angles, y=shift_cylinder, color='black', linestyle=':', label='cylinder', legend=False)
plt.ylabel(r'freq. shift [ppm]')
plt.xlabel(r'$\theta$ [$\degree$]')
plt.title('frequency shift')
plt.ylim(-0.15, 0.25)

# decay vs angle
angles = fids.angle.unique()
times = fids.time.unique()
timepoint = times[np.argmin(abs(times - 10))]
unitcorr = 1e-3 * 2 * np.pi 


# water
data = fids[(fids.region=='water')&(fids.time == timepoint)].copy()
data['decay'] = -np.log(abs(data.fid))

data['cfreq'] = 2 * np.pi * np.sin(data.angle / 180 * np.pi )**2 * (CHI_F - CHI_W) * ppmfreq * 1e-3
data['pred'] = 1/4 * data.fraction * data.cfreq**2 * timepoint**2 * unitcorr

ax = axes[1, 1]
plt.sca(ax)
sns.lineplot(data=data, x='angle', y='decay', hue='fraction', palette='flare_r', legend=False)
if not 'PERLIN' in info['OBJECT']:
    sns.lineplot(data[data.nexp==0], x='angle', y='pred', hue='fraction', palette='flare_r', linestyle=':', legend=False)
plt.xlabel(r'$\theta$ [$\degree$]')
plt.ylabel(r'$\eta_w (t=10ms)$')
plt.title('water decay')
plt.ylim(-0.05, 0.7)

# fat
data = fids[(fids.region=='fat')&(fids.time == timepoint)].copy()
data['decay'] = -np.log(abs(data.fid))
# data['true_frac'] = data.fraction.map({key[1]: np.round(mask.mean(), 2) for key, mask in masks.items()}).astype('category')

ax = axes[1, 2]
ax.sharex(axes[1, 1])
ax.sharey(axes[1, 1])
plt.sca(ax)
sns.lineplot(data=data, x='angle', y='decay', hue='fraction', palette='flare_r', legend='full')
plt.xlabel(r'$\theta$ [$\degree$]')
plt.ylabel(r'$\eta_f (t=10ms)$')
plt.title('fat decay')
legend_handles = ax.legend_.legend_handles
[h.set_label(f) for f,h in zip(fraction_pc, legend_handles)]
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', title='fat fraction', handles=legend_handles)



#
# signal decay  
max_time = 50
fids = fids[fids.time < max_time]

# water 
data = fids[fids.region=='water'].copy()
data['logfid'] = -np.log(data['fid']) / np.maximum(np.sin(data['angle'] / 180 * np.pi)**4, 1e-10)
data['time2'] = data['time']**2
# quadratic rate
qrate = data[(data.angle==15)&(data.time==times[1])].groupby('fraction', as_index=True)['logfid'].mean() #nth(1)
qrate /= times[1]**2
qrate.name ='qrate'
data = pd.merge(data, qrate, on=['fraction'])
data['qline'] = data['qrate'] * data['time2']

# linear
fraction = data.fraction.unique()
tt = np.array([fids.time.max() - 5, fids.time.max() + 5])
angle = angles[3:]
freq = 2 * np.pi * np.sin(angle / 180 * np.pi )**(-2) * (CHI_F - CHI_W) * ppmfreq * 1e-3 * unitcorr**0.5
slope = pd.concat([pd.Series(freq * f, name='slope', index=pd.Index(angle, name='angle')) for f in fraction], axis=0, keys=fraction, names=['fraction'])
lines = pd.concat([slope * t for t in tt], axis=0, keys=tt, names=['time']).rename('line').reset_index()

ax1 = axes[0, 1]
plt.sca(ax1)
sns.lineplot(data=data[(data.angle==0)&(data.nexp==0)], x='time', y='qline', hue='fraction', palette='flare_r', linestyle=':', errorbar=None, legend=False)
if info['OBJECT'] == 'RCYLINDERS':
    sns.lineplot(lines.loc[lines.angle==90], x='time', y='line', hue='fraction', palette='flare_r', dashes=(3,1), legend=False) 
for angle, df in data.groupby('angle'):
    alpha = angle/90
    sns.lineplot(data=df, x='time', y='logfid', hue='fraction', palette='flare_r', alpha=alpha, errorbar=None, legend=False)
plt.axvline(timepoint, color='gray', linestyle='--')
plt.grid(False, axis='y')
ax1.set_yticklabels([])
plt.ylabel(r'$\eta_w(t)/\sin(\theta)^4$')
plt.xlabel(r'$t\ [ms]$')
plt.title('water decay')

# fat
data = fids[fids.region=='fat'].copy()
data['logfid'] = -np.log(data['fid']) / np.maximum(np.sin(data['angle'] / 180 * np.pi)**4, 1e-10)
data['time2'] = data['time']**2
# quadratic rate
qrate = data[(data.angle==15)&(data.time==times[1])].groupby(['fraction'], as_index=True)['logfid'].mean() #nth(1)
qrate /= times[1]**2
qrate.name ='qrate'
data = pd.merge(data, qrate, on=['fraction'])
data['qline'] = data['qrate'] * data['time2']
data['lline'] = data['qrate'] * data['time']

ax2 = axes[0, 2]
plt.sca(ax2)
ax2.sharey(ax1)
sns.lineplot(data=data[(data.angle==0) & (data.nexp==0)], x='time', y='qline', hue='fraction', palette='flare_r', linestyle=':', errorbar=None, legend=False)
for angle, df in data.groupby('angle'):
    alpha = angle/90
    sns.lineplot(data=df, x='time', y='logfid', hue='fraction', palette='flare_r', alpha=alpha, errorbar=None, legend=True)
plt.axvline(timepoint, color='gray', linestyle='--')
plt.grid(False, axis='y')
plt.ylabel(r'$\eta_f(t)/\sin(\theta)^4$')
plt.xlabel(r'$t\ [ms]$')
plt.title('fat decay')

# legend
[h.set_label(f) for f,h in zip(fraction_pc, legend_handles)]
angles_subset = [angles[1], angles[len(angles)//2], angles[-1]]
fflines = [plt.Line2D([],[], label=rf'$\theta={angle}$', linestyle='-', color='k', alpha=angle/90) for angle in angles_subset]
legend_handles.append(plt.Line2D([],[], linestyle=''))
legend_handles.extend(fflines)
sns.move_legend(ax2, loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', title='fat fraction', handles=legend_handles)


# done

if SHOW:
    plt.show()
else:
    # fig.savefig(exp_name + '.svg')
    fig.savefig(exp_name + '.png', dpi=500)


