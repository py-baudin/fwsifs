"""
Phase unwrapping algorithm

Dymerska, Barbara, Korbinian Eckstein, Beata Bachrata, Bernard Siow, Siegfried Trattnig, Karin Shmueli, et Simon Daniel Robinson. 
« Phase Unwrapping with a Rapid Opensource Minimum Spanning Tree Algorithm (ROMEO) ». 
Magnetic Resonance in Medicine 85, nᵒ 4 (2021): 2294‑2308. 
https://doi.org/10.1002/mrm.28563.

"""

import numpy as np
import numba as nb
import heapq

import logging

NAX = np.newaxis
LOGGER = logging.getLogger(__name__)


def unwrap(phase, *, mask=None, mag=None):
    """phase unwrapping"""

    # get edges
    edges = get_edges(phase.shape)  # , mask=mask)

    # compute edge weights, values between 0 and 1
    LOGGER.info("Compute spacial coherence weights")
    weights1 = spacial_coherence_weights(phase, edges)

    weights3 = 1.0
    if mag is not None:
        LOGGER.info("Compute magnitude coherence weights")
        weights3 = magnitude_coherence_weights(mag, edges)  # , mask)

    # compute quality map (product of weights)
    qmap = weights1 * weights3

    # round qmap between 0 and 255 (8 bits uint)
    # (0 is for no connection)
    costs = np.round((1 - qmap) * 255).astype("uint8")
    costs[costs == 0] = 1
    costs[edges < 0] = 0

    # init mask
    init_mask = bbox_mask(phase.shape)
    if mask is not None:
        init_mask &= mask
    qmin = np.min(qmap, axis=1)
    init = np.argmax(qmin * init_mask.flat)

    # init arrays with correct dtypes
    edges = edges.astype("int32")
    init = np.int32(init)

    # start unwrapping (normalize phase range in [-0.5, 0.5])
    LOGGER.info("Start unwrapping")
    phi = phase / 2 / np.pi
    uphi, order = unwrap_phase(phi, edges, costs, init)

    # return unwrapped phase
    uphase = uphi * 2 * np.pi
    return uphase


@nb.njit(cache=True)
def unwrap_phase(phi, edges, costs, init):
    """phase unwrapping (inplace)
    Expects phase phi normalized between -0.5 and 0.5.
    """
    nneigh = edges.shape[1]

    # initialize heap queue
    queue = [(costs[init, i], init, edges[init, i]) for i in range(nneigh)]
    heapq.heapify(queue)

    # visited mask
    visited = np.zeros(edges.shape[0], dtype=nb.boolean)
    visited[init] = True
    phi.flat[init] = np.mod(phi.flat[init] + 0.5, 1) - 0.5

    order = np.zeros(phi.shape, dtype=np.int64)
    index = 0
    while queue:
        # get edge
        _, v1, v2 = heapq.heappop(queue)
        if visited[v2]:
            # v1 is always visited
            continue

        # update phi inplace
        phi.flat[v2] += np.round(phi.flat[v1] - phi.flat[v2])
        visited[v2] = True
        order.flat[v2] = index + 1
        index += 1

        # add neighbors
        for i in range(nneigh):
            v, w = edges[v2, i], costs[v2, i]
            if w == 0 or visited[v]:
                continue
            heapq.heappush(queue, (w, v2, v))

    return phi, order


def get_edges(shape, mask=None):
    """get edges vertices"""
    ndim = len(shape)
    size = np.prod(shape)
    steps = [int(np.prod(shape[d + 1 :])) for d in range(ndim)]

    # find edges
    indices = np.arange(size)

    edges = []
    for d in range(ndim):
        for s in (-1, 1):
            coord = np.mod(indices // steps[d], shape[d]) + s
            edges_ds = indices + s * steps[d]
            edges_ds[(coord < 0) | (coord >= shape[d])] = -1
            if mask is not None:
                edges_ds[~mask.flat[edges_ds]] = -1
            edges.append(edges_ds)
    edges = np.stack(edges, axis=1)
    return edges


def spacial_coherence_weights(phase, edges):
    phasor = np.exp(1j * phase)
    weights = 1 - np.abs(np.angle(phasor.flat[edges] / phasor.ravel()[:, NAX])) / np.pi
    weights[edges < 0] = 0
    return weights


def temporal_coherence_weights(volumes, edges, mask=None): ...


def magnitude_coherence_weights(mag, edges, mask=None):
    minmag = np.minimum(mag.flat[edges], mag.ravel()[:, NAX])
    maxmag = np.maximum(mag.flat[edges], mag.ravel()[:, NAX])
    weights = (minmag / (maxmag + 1e8 * (maxmag < 1e-8))) ** 2
    if mask is not None:
        weights[mask.ravel() == 0] = 1
    weights[edges < 0] = 0
    return weights


def bbox_mask(shape, fraction=1 / 3):
    """return mask of central region"""
    ndim = len(shape)
    fraction = np.ones(ndim) * fraction
    npix = [int(shape[i] * fraction[i]) for i in range(ndim)]
    bbox = [slice(npix[i], shape[i] - npix[i]) for i in range(ndim)]
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(bbox)] = True
    return mask


# def spacial_centering_weights(phase, edges, mask):
#     """ windowing weights (decrease near the border of the image) """
#     spacing = np.array(getattr(phase, 'spacing', [1] * phase.ndim))
#     center = np.array(phase.shape) / 2
#     coords = np.indices(phase.shape).reshape(phase.ndim, -1)
#     dist = np.linalg.norm((coords - center[:, np.newaxis]) * spacing[:, np.newaxis], axis=0)
#     # dist = np.linalg.norm((coords - center[:, np.newaxis]), axis=0)
#     dist = dist.reshape(phase.shape) / dist.max()
#     dist[mask] = 0
#     weigths = (1 - dist.flat[edges]) * (edges >= 0)
#     return weigths


# def fix_masked_phase(phase, mask, size=31):
#     ndim = phase.ndim
#     mask = mask > 0
#     spacing = np.array(getattr(phase, 'spacing', [1,]*ndim))
#     size = np.round(size * np.min(spacing) / spacing).astype(int)
#     fphase = phase * 0
#     fphase[:] = uniform_filter(mask * phase, size=size)[:]
#     fphase[mask] = phase[mask]
#     fmask = uniform_filter(1.0 * mask, size=size)
#     valid = (mask < 1) & (fmask > 1e-5)
#     fphase[valid] /= fmask[valid]
#     return fphase

# def uniform_filter(im, size=5):
#     ndim = im.ndim
#     size = np.ones(ndim, dtype=int) * size
#     for d in range(ndim):
#         # pad image
#         padding = [(size[d], size[d]) if i == d else (0, 0) for i in range(ndim)]
#         padim = np.moveaxis(np.pad(im, padding), d, -1)
#         # convolve
#         ker = np.ones(size[d]) / size[d]
#         padim = np.convolve(padim.ravel(), ker, mode='same').reshape(padim.shape)
#         # recover im
#         slices = [slice(size[d], -size[d]) if i == d else slice(None) for i in range(ndim)]
#         im = np.moveaxis(padim, -1, d)[tuple(slices)]
#     return im
