import os
import abc
from types import SimpleNamespace
import numpy as np
from . import pmath


NAX = np.newaxis
DEBUG = os.environ.get('OPTIM_DEBUG', False)



def nlls(obj, *, method='lm', maxiter=100, maxmoving=0, ftol=1e-8, gtol=1e-7, opts=None, disp=False, callback=None):
    """ Solver for non-linear least-squares
    
    todo:
    - trust region update (LM method)
    - hybrid step with secant equation (Dennis, Gay, and Welsch or BFGS) 
    - `mask` option in fun and jac


    Nocedal, Jorge, et Stephen J. Wright. Numerical Optimization. Second edition. Springer Series in Operations Research and Financial Engineering. New York, NY: Springer, 2006.

    """

    res = SimpleNamespace(**{'nit': 0, 'nfev': 0, 'njev': 0, 'msg': 'pending', 'success': False})
    opts = opts or {}

    init = tovec(obj.init())
    obj.update(init)
    
    res.method = method
    res.x = init
    res.f = tovec(obj.fun(init))
    res.c = 0.5 * vecdot(res.f, res.f).real
    res.cost = np.sum(res.c)
    res.nfev += 1
    res.jac = 1

    moving = np.ones(len(res.x), dtype=bool)
    if disp: print(f'Solving with {method}'); disp_it(res, moving)

    for i in range(maxiter):
        res.nit = i + 1
        if method == 'sd':
            # Steepest descent step
            x, f, c, u = steepest_descent(obj, res, ftol=ftol, **opts)
            
        if method == 'gn':
            # Gauss Newton step
            x, f, c, u = gauss_newton(obj, res, ftol=ftol, **opts)
        
        elif method == 'lm':
            # Levenberg Marquardt step
            x, f, c, u = levenberg_marquardt(obj, res, gtol=gtol, **opts)

        # total cost
        cost = np.sum(c)
        if cost > res.cost:
            res.success = False
            res.msg = 'Total cost increased.'
            break

        # update gradient
        res.x = x
        res.f = f
        res.c = c
        res.cost = cost
        moving &= u
        
        # update objective
        fu = obj.update(x)
        if fu is not None:
            res.f = fu

        if disp: disp_it(res, moving)
        if callback:
            callback(res)

        # stop
        if np.sum(moving) <= maxmoving:
            res.success = True
            res.msg = 'Stopping criterion reached.'
            break

    else:
        res.msg = 'Maximum number of iterations reached.'
    

    if disp: disp_res(res)
    return res



class Objective(abc.ABC):
    """ abstract objective """

    @classmethod
    def basic(cls, init, fun, jac, *, update=None):
        """ Basic initializer """
        _init = lambda self: np.array(init)
        _fun = lambda self, x: fun(x)
        _jac = lambda self, x: LinearOperator(jac(x))
        def _hess(self, x):
            J = jac(x)
            # H = hermitian(J) @ J
            H = matmat(J, J, transpose=True, conj=True)
            return LinearOperator(J), LinearOperator(H)
        _update = lambda self, x: (update(x) if callable(update) else None) 
        obj = type('MyObjective', (cls,), dict(init=_init, fun=_fun, jac=_jac, hess=_hess, update=_update))
        return obj()
    
    @abc.abstractmethod
    def init(self):
        pass
    @abc.abstractmethod
    def fun(self, x):
        pass
    @abc.abstractmethod
    def jac(self, x):
        pass
    def hess(self, x):
        raise NotImplementedError()
    def update(self, x):
        pass


class LinearOperator:
    """ linear operator """
    def __init__(self, arr=None, *, dot=None, hdot=None, solve=None):
        if arr is not None:
            op = tomat(arr)
            dot = lambda x: matvec(op, x)   
            hdot = lambda b: matvec(op, b, transpose=True, conj=True).real # real
            solve = lambda x, mu: linsolve(op.real, x, mu=mu) # real
            self._op = op
        self._dot = dot
        self._hdot = hdot
        self._solve = solve
    
    def dot(self, b):
        if self._dot is None:
            raise NotImplementedError()
        return self._dot(b)
    
    def hdot(self, b):
        if self._hdot is None:
            raise NotImplementedError()
        return self._hdot(b)
    
    def solve(self, b, *, mu=None):
        if self._solve is None:
            raise NotImplementedError()
        return self._solve(b, mu=mu)


#
# functions

def steepest_descent(obj, res, *, ftol=1e-8, ls_maxiter=10, **opts):
    """ steepest descent """
    J = obj.jac(res.x) # M x N
    res.njev += 1

    g = tovec(J.hdot(res.f))

    # linesearch
    x, f, c, u = linesearch(obj.fun, res, g, ftol=ftol, maxiter=ls_maxiter, **opts)
    return x, f, c, u


def gauss_newton(obj, res, ftol=1e-8, ls_maxiter=10, **opts):
    """ Gauss Newton """
    J, H = obj.hess(res.x) # M x N
    res.njev += 1
    g = tovec(J.hdot(res.f))
    v = tovec(H.solve(g))

    # line search
    x, f, c, u = linesearch(obj.fun, res, v, g, ftol=ftol, maxiter=ls_maxiter, **opts)
    return x, f, c, u


def levenberg_marquardt(obj, res, *, eta=1/8, mu=1, gtol=1e-7):
    """
    Fan, Jinyan, et Jianyu Pan. « A note on the Levenberg–Marquardt parameter ». 
    Applied Mathematics and Computation 207, nᵒ 2 (15 janvier 2009): 351‑59. https://doi.org/10.1016/j.amc.2008.10.056.
    """
    one = np.array(1)
    x, f = res.x, res.f
    c0 = 0.5 * vecdot(f, f).real * one

    # compute step p
    J, H = obj.hess(res.x)
    res.njev += 1
    g = tovec(J.hdot(f))
    rho = np.minimum(vnorm(f), vnorm(g)) #* (c0.size / x.size)

    mu = getattr(res, 'lm_mu', mu) * np.ones(c0.shape)
    p = - tovec(H.solve(g, mu=mu * rho))

    # test updated solution
    fp = tovec(obj.fun(x + p))
    res.nfev += 1
    cp = 0.5 * vecdot(fp, fp).real * one

    # assess progress
    Jp = tovec(J.dot(p))
    pred = - vecdot(g, p).real - 0.5 * vecdot(Jp, Jp).real
    obs = c0 - cp
    
    # increase regularization if progress is slow
    r1 = obs < 1/4 * pred
    mu[r1] *= 2

    # decrease regularization if progress is fast
    r2 = (obs > 3/4 * pred) & (mu >= 1e-8)
    mu[r2] = mu[r2] * 1/4
    
    # update x
    rn = obs > eta * pred
    x[rn] += p[rn]
    cp[~rn] = c0[~rn]
    fp[~rn] = f[~rn]
    p[~rn] = 0

    res.lm_mu = mu
    res.lm_p = p
    updated = rho > gtol

    if DEBUG: print(f'update: {np.mean(updated)*100:.0f}%, increase mu: {np.mean(r1)*100:.0f}%, decrease mu: {np.mean(r2)*100:.0f}%, mu={np.mean(mu)}')    
    return x, fp, cp, updated




def linesearch(fun, res, v, j=None, *, step=1, ftol=1e-8, alpha=0.5, beta=0.7, gamma=0, h=1, maxiter=100):
    """ line search with two way backtracking 
    
    Truong, Tuyen Trung, et Hang-Tuan Nguyen. 
    « Backtracking Gradient Descent Method and Some Applications in Large Scale Optimisation. Part 2: Algorithms and Experiments ». 
    Applied Mathematics & Optimization 84, nᵒ 3 (décembre 2021): 2557‑86. https://doi.org/10.1007/s00245-020-09718-8.

    """
    # backtracking bounds
    if j is None:
        j = v
    armijo = vecdot(v, j).real
    ubound = step * np.maximum(1, h / (vnorm(j) + 1e-10))
    shape = armijo.shape

    # init
    c0 = 0.5 * vecdot(res.f, res.f).real
    direction = None
    moving = np.ones(shape, dtype=bool)
    step = getattr(res, 'lns_step', step) * np.ones(shape)
    p_prev = getattr(res, 'lns_p', 0)

    for i in range(maxiter):
        # parameter update
        p = v * step[..., NAX]
        if gamma:
            # add momentum
            p += gamma * p_prev
        x = res.x - p

        # step eval
        f = tovec(fun(x)) # TODO: compute only moving values
        res.nfev += 1
        cp = 0.5 * vecdot(f, f).real

        # check Armijo’s condition
        valid = cp <= c0 - alpha * step * armijo

        if direction is None:
            direction = 2 * valid - 1
        else:
            # largest step is found
            moving[valid & ((direction < 0) | (step / beta > ubound))] = False
            # moving[valid & (direction < 0)] = False

            # step too large: recover previous step
            recover = (~valid) & (direction > 0)
            if np.any(recover):
                moving[recover] = False
                step[recover] *= beta
                f[recover] = _f[recover]
                p[recover] = _p[recover]
                x[recover] = res.x[recover] - _p[recover]

        # step upper bound is reached
        # moving[(direction > 0) & (step / beta > ubound)] = False

        if np.all(~moving):
            # all steps frozen
            break

        # update moving steps
        step[moving & valid] /= beta
        step[moving & ~valid] *= beta

        # store current parameters
        _p, _f = p, f

    res.lns_step = step
    res.lns_p = p
    c = 0.5 * vecdot(f, f).real
    updated = abs(c0 - c) > ftol * np.maximum(1, abs(c0))
    return x, f, c, updated


# utilities

def tovec(arr):
    return np.atleast_2d(arr)

def tomat(arr):
    arr = np.asanyarray(arr)
    dims = tuple(np.arange(max(3 - arr.ndim, 0)))
    return np.expand_dims(arr, dims) if dims else arr


hermitian = pmath.hermitian
vecdot = pmath.vecdot # np.vecdot
matvec = pmath.matvec
matmat = pmath.matmat
linsolve = pmath.linsolve

# def matvec(A, v):
#     """ matrix vector product"""
#     return np.matmul(A, v[..., NAX])[..., 0]

def vnorm(v, B=None):
    """ vector norm (optionally weighted)"""
    v1, v2 = v, v
    if B is not None:
        v2 = matvec(B, v2)
    return vecdot(v1, v2).real**0.5

# def solve(H, b, *, mu=None, maxcond=1e10):
#     if mu is not None:
#         H = H.copy()
#         inds = np.arange(H.shape[-1])
#         H[..., inds, inds] += np.asanyarray(mu)[..., NAX]
#     try:
#         return np.linalg.solve(H, b[..., NAX])[..., 0]
#     except np.linalg.LinAlgError as exc:
#         pass
#     valid = np.linalg.cond(H) < maxcond
#     x = np.zeros_like(b)
#     if np.any(valid):
#         x[valid] = np.linalg.solve(H[valid], b[valid, ..., NAX])[..., 0]
#     return x    


# 
# display

def disp_it(res, moving, max_n=3):
    print(f'iteration {res.nit}: f(x) = {np.sum(res.c)}, num. moving = {np.sum(moving)}')

def disp_res(res, max_n=3):
    print('-' * 50)
    print(f'Success: {res.success}')
    print(f'Message: {res.msg}')
    print(f'cost = {np.sum(res.cost)}')
    # print(f'x = {res.x[:max_n]}')
    print(f'Num. iter: {res.nit}')
    print(f'Num. function ev.: {res.nfev}')
    print(f'Num. jacobian ev.: {res.njev}')