'''
run microstructure simulation 

Generate experiment file: 
example: `objRCYLINDERS_3_nff5_img1001_bin_nexp20.pickle`
(objects=random cylinders, size=3, num fraction=5, image size=1001, num. experiments=20)

usage:
python microstructure_gen.py

example: run simulation for random cylinders
set `OBJECT = 'RCYLINDERS'`
(cf. below)



'''

import time
import numpy as np
import numexpr as ne
import pandas as pd

# requires perlinnoise package
# at https://github.com/py-baudin/perlinnoise
try:
    import perlinnoise
except ImportError:
    print('Perlin noise simulation requires `perlinnoise` package')
    print('download from: `https://github.com/py-baudin/perlinnoise`')


# object shape
# OBJECT = 'SPHERE' 
# OBJECT = 'CYLINDER'
# OBJECT = 'RSPHERES' 
# OBJECT = 'AGGSPHERES' 

# OBJECT = 'RCYLINDERS' 
# OBJECT = 'AGGCYLINDERS' 
OBJECT = 'PERLIN2D'

# object's size
SIZE = 3

# fat fractions
FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]

# hard or soft mask
FUNC = 'bin'
# FUNC = 'sigm'

# tested angles
ANGLES = np.arange(0, 91, 15)

# simulation dimensions
SHAPE = np.array([1001, 1001, 1])
DIMS = np.array([1, 1, 1]) # mm
SPACING = np.ones(len(DIMS)) # DIMS / SHAPE

# number of experiments
NEXP = 20

# experiment tag
info = {'NEXP': NEXP, 'FRACTIONS': FRACTIONS, 'ANGLES': ANGLES, 'OBJECT': OBJECT, 'SHAPE': SHAPE, 'FUNC': FUNC}
exp_name = f'obj{OBJECT}_{SIZE}_nff{len(FRACTIONS)}_img{SHAPE[0]}_{FUNC}_nexp{NEXP}'

# constants
PI = np.pi
GAMMA = 42580.0 # kHz/T
B0 = 2.89 # T
g = GAMMA * B0 * 1e-6 * 2 * np.pi # 1e-6 rad.kHz

# chi (ppm)
CHI_W = -9.05
# CHI_F = -7.79 #  Fat, Hoptkins 1997
CHI_F = -8.44 # EMCL, Boesch, 1997

# times (ms)
# TIMES = np.linspace(0, 20, 100) 
TIMES = np.linspace(0, 50, 201)
ECHOS = np.array([2.22, 5.42, 8.62, 11.82, 15.02, 18.22]) # 6pt echo times

# object generator
kwargs = {}
if OBJECT == 'SPHERE':
    objects = 'Sphere'
    def generate_object(ff):
        radius = (3/4 / PI * ff)**(1/3)
        return make_sphere(radius)[0]
elif OBJECT == 'CYLINDER':
    objects = 'Cylinder'
    def generate_object(ff):
        radius = (ff / PI)**(1/2)
        return make_cylinder(radius)[0]
elif OBJECT == 'RSPHERES':
    objects = f'Rand. spheres'
    def generate_object(ff):
        radius = SIZE * 1e-2
        return pack_objects(make_sphere, ff, radius)[0]
elif OBJECT == 'AGGSPHERES':
    objects = f'Aggr. spheres'
    def generate_object(ff):
        radius = SIZE * 1e-2
        return pack_aggregated_objects(make_sphere, ff, radius)[0]
elif OBJECT == 'RCYLINDERS':
    objects = f'Rand. cylinders'
    def generate_object(ff):
        radius = SIZE * 1e-2
        return pack_objects(make_cylinder, ff, radius)[0]
elif OBJECT == 'AGGCYLINDERS':
    objects = f'Aggr. cylinders'
    def generate_object(ff):
        radius = SIZE * 1e-2
        return pack_aggregated_objects(make_cylinder, ff, radius)[0]
elif OBJECT == 'PERLIN2D':
    objects = '2D Perlin'
    def generate_object(ff):
        num = int(1 / (2 * SIZE * 1e-2))
        return make_perlin_2d(ff, num)[0]
elif OBJECT == 'PERLIN3D':
    objects = '3D Perlin'
    def generate_object(ff):
        num = int(1 / (2 * SIZE * 1e-2))
        return make_perlin_3d(ff, num)[0]
    

# volume coordinates
indices = np.moveaxis(np.indices(SHAPE), 0, -1)
coords = (indices - SHAPE//2) / np.maximum(SHAPE - 1, 1)
zeros = np.zeros(SHAPE)

def sigmoid(arr, r=1):
    """ soft thresholding """
    return ne.evaluate('1 / (1 + exp(- maximum(r * arr, -1e2)))')

# volume functions
def make_sphere(radius, *, loc=(0, 0, 0)):
    loc = np.asarray(loc)
    dists = ne.evaluate('(((coords - loc + 0.5) % 1) - 0.5)**2')
    dist = np.sum(dists, axis=-1)
    if FUNC == 'bin':
        obj = ne.evaluate('dist <= radius**2')
    elif FUNC == 'sigm':
        tol = radius * 1e5
        obj = sigmoid(ne.evaluate('radius - dist**0.5'), r=tol)
    return obj, dists
    
def make_cylinder(radius, *, loc=(0, 0, 0)):
    global coords
    loc = np.asarray(loc)[:2]
    coords = coords[..., :2]
    dists = ne.evaluate('(((coords - loc + 0.5) % 1) - 0.5)**2')
    dist = np.sum(dists, axis=-1)
    if FUNC == 'bin':
        obj = ne.evaluate('dist <= radius**2')
    elif FUNC == 'sigm':
        tol = radius * 1e5
        obj = sigmoid(ne.evaluate('radius - dist**0.5'), r=tol)
    dists[obj > 0.5] = 0
    return obj, dists

def make_blob(radius, *, loc=(0, 0, 0)):
    global coords
    loc = np.asarray(loc)[:2]
    coords = coords[..., :2]
    relpos = ne.evaluate('((coords - loc + 0.5) % 1) - 0.5')
    dist2 = np.sum(relpos**2, axis=-1)
    # angle for each coordinate
    angles = np.atan(relpos[..., 1] / relpos[..., 0])
    swap1 = (relpos[..., 0] < 0) & (angles < 0)
    swap2 = (relpos[..., 0] < 0) & (angles > 0)
    angles += np.pi * swap1
    angles -= np.pi * swap2
    iszero = np.isclose(relpos[..., 1], 0)
    angles[iszero] = 0.5 * (np.sign(relpos[iszero, 0]) - 1) * np.pi
    # generate polygon
    nbin = 5
    bins = np.ones(nbin) + np.random.uniform(-1, 1, nbin) * 1e-1
    bins = np.cumsum(np.r_[0, bins])
    bins = 2 * np.pi * bins / bins[-1] - np.pi
    width = np.diff(bins)
    values = np.random.uniform(0.5, 2, nbin)
    # bin of each angle
    idx = np.minimum(np.digitize(angles, bins) - 1, nbin - 1)
    for i in range(nbin):
        # interpolate between each radius
        select = idx==i
        coef1 = (bins[i + 1] - angles[select]) / width[i]
        coef2 = (angles[select] - bins[i]) / width[i]
        dist2[select] *= values[i] * coef1 + values[(i + 1) % nbin] * coef2
    obj = ne.evaluate('dist2 <= radius**2')
    dist2[obj > 0.5] = 0
    return obj, relpos**2

def make_perlin_2d(fraction, num):
    import perlinnoise
    im = perlinnoise.perlin_noise(SHAPE[:-1], num)

    thresh = np.percentile(im, 100 - int(fraction * 100))
    if FUNC == 'bin':
        obj = np.stack([im > thresh] * SHAPE[-1], axis=-1)
    elif FUNC == 'sigm':
        tol = radius * 1e5
        obj = np.stack([sigmoid(thresh - im, r=tol)] * SHAPE[-1], axis=-1)
    return obj, None


def make_perlin_3d(fraction, num):
    import perlinnoise
    im = perlinnoise.perlin_noise(SHAPE, num)
    thresh = np.percentile(im, 100 - int(fraction * 100))
    if FUNC == 'bin':
        obj = im > thresh
    elif FUNC == 'sigm':
        tol = radius * 1e2
        obj = sigmoid(thresh - im, r=tol)
    return obj, None

def pack_objects(generator, fraction, radius, *, mindist=-1/10):
    canvas = zeros.copy()
    dists = np.inf * np.ones(SHAPE)
    num = 0
    while True:
        valid = ne.evaluate('dists >= radius * (1 + mindist)')
        nvalid = valid.sum()
        if not nvalid:
            print(f'Warning: reached packing limit ({canvas.sum() / canvas.size:.2%})')
            break
        index = np.random.randint(nvalid)
        loc = coords[valid][index]
        obj, dists_ = generator(radius, loc=loc)
        canvas += obj
        num += 1
        if canvas.sum() > fraction * canvas.size:
            print(f'Packed {num} objects')
            break
        idist = np.sum(dists_, axis=-1)
        dists = ne.evaluate('minimum(dists, maximum(idist**0.5 - radius, 0))')
    canvas = np.minimum(canvas, 1)
    return canvas, dists


# attract=1e2, mindist=-1/5
def pack_aggregated_objects(generator, fraction, radius, *, mindist=0, attract=1e2, restart=1/200):
    canvas = zeros.copy()
    dists = np.inf * np.ones(SHAPE)
    pdists = 1e2 * np.ones(SHAPE)
    a = attract * np.ones(3)
    num = 0
    while True:
        if restart and (np.random.uniform() < restart):
            pdists[:] = 1e2
        valid = ne.evaluate('dists >= radius * (1 + mindist)')
        nvalid = valid.sum()
        if not nvalid:
            print(f'Warning: reached packing limit ({canvas.sum() / canvas.size:.2%})')
            break
        pdist = pdists[valid]
        prob = ne.evaluate('exp(-pdist)')
        prob /= prob.sum()
        index = np.random.choice(nvalid, p=prob)
        loc = coords[valid][index]
        obj, dists_ = generator(radius, loc=loc)
        canvas += obj
        num += 1
        if canvas.sum() > fraction * canvas.size:
            print(f'Packed {num} objects')
            break
        idist = np.sum(dists_, axis=-1)
        adist = attract**2 * idist if np.isscalar(attract) else np.dot(dists_, a**2)
        dists = ne.evaluate('minimum(dists, maximum(idist**0.5 - radius, 0))')
        pdists = ne.evaluate('minimum(pdists, maximum(adist, 0))')
        
    canvas = np.minimum(canvas, 1)
    return canvas, dists


# dipole kernel
freqs = [np.fft.fftfreq(SHAPE[ax], SPACING[ax]) for ax in range(3)]
k = np.stack([ki[idx] for idx, ki in zip(np.indices(SHAPE), freqs)])

def dipole_kernel(theta):
    theta_rad = theta * np.pi / 180
    vec = np.array([np.sin(theta_rad), 0, np.cos(theta_rad)])
    k[:, 0, 0, 0] = 1
    K = 1 / 3 - (np.dot(k.T, vec)**2).T / np.sum(k * k, axis=0)
    K.flat[0] = 0
    
    return K

def fftconvolve(im, ker):
    im_f = np.fft.ifftn(ker * np.fft.fftn(im))
    return im_f.real

kernels = {angle: dipole_kernel(angle) for angle in ANGLES}

#
# run experiments 
print(f'Running experiments')
print(f'Simulated objects: {OBJECT}')

tic1 = time.time()
stats, fids, spectra, objects = [], [], [], []
for nexp in range(NEXP):
    print(f'experiment: {nexp + 1} / {NEXP}')

    for ff in FRACTIONS:

        # generate object 
        print(f'\tGenerate object (ff={ff})')
        obj = generate_object(ff)

        # make chimap and mask
        chimap = obj * (CHI_F - CHI_W) + CHI_W
        mask = obj > 0.5

        # store objects
        if nexp == 0:
            objects.append({'nexp': nexp, 'fraction': ff, 'mask': mask, 'chimap': chimap})

        for angle in ANGLES:
            kernel = kernels[angle]

            print(f'\t\tfrequency map (angle={angle})')
            freqmap = fftconvolve(chimap, kernel)

            # FID signal decay
            # print('\tsignal decay')
            fid = []
            for t in TIMES:
                signal = ne.evaluate('exp(1j * freqmap * t * g)')
                fid.append((np.mean(signal[~mask]), np.mean(signal[mask])))
                # fid.append((np.mean(signal[mask]), np.mean(signal[~mask])))
            fid = np.array(fid).T

            # spectrum
            # print('\tspectra')
            NFFT = 256
            freqs_ppm = np.linspace(-0.5, 0.5, NFFT)
            spectr = np.stack([
                np.histogram(freqmap[~mask], bins=freqs_ppm)[0] / mask.size,
                np.histogram(freqmap[mask], bins=freqs_ppm)[0] / mask.size,
            ])

            # store simulations results
            stats.append(dict(
                nexp=nexp,
                fraction=ff,
                angle=angle, 
                mean_w=np.mean(freqmap[~mask]),
                mean_f=np.mean(freqmap[mask]),
                var_w=np.var(freqmap[~mask]),
                var_f=np.var(freqmap[mask]),
                shift=np.mean(freqmap[mask]) - np.mean(freqmap[~mask]),
            ))

            n = len(TIMES)
            fids.extend([dict(
                nexp=nexp,
                fraction=ff,
                angle=angle,
                time=TIMES[i%n],
                region='water' if i < n else 'fat',
                fid=abs(fid[1 * (i >= n), i % n]),
            ) for i in range(2 * n)])

            n = len(freqs_ppm) - 1
            spectra.extend([dict(
                nexp=nexp,
                fraction=ff,
                angle=angle,
                freq=freqs_ppm[i % n],
                region='water' if i < n else 'fat',
                spectrum=abs(spectr[1 * (i >= n), i % n]) / np.max(abs(spectr[1 * (i >= n)])),
            ) for i in range(2 * n)])
        
    
tic2 = time.time()
print(f'Done simulation ({tic2 - tic1:.1f}s)')

stats = pd.DataFrame(stats)
fids = pd.DataFrame(fids)
spectra = pd.DataFrame(spectra)

# store results
import pickle
with open(exp_name + '.pickle', 'wb') as fp:
    pickle.dump({'info': info, 'stats': stats, 'fids': fids, 'spectra': spectra, 'objects': objects}, fp)
