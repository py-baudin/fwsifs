import os
import numpy as np

NAX = np.newaxis

# not using numba for now
PMATH_NUMBA = bool(int(os.environ.get('PMATH_NUMBA', False)))

# utilities

def vecdot(a, b):
    """ vector product """
    return vecvec(a, b, conj=True)

def hprod(A, b):
    """ matrix vector product with conjugated transpose """
    return matvec(A, b, transpose=True, conj=True)

def hermitian(A):
    """ hermician transpose """
    return transpose(A, conj=True)



# default functions

def _matvec(A, b, *, transpose=False, conj=False):
    """ matrix vector product """
    if transpose:
        A = A.mT
    if conj:
        A = A.conj()
    return np.matmul(A, b[..., NAX])[..., 0]

def _vecvec(a, b, *, conj=False):
    """ vector vector product """
    if conj:
        a = a.conj()
    return np.sum(a * b, axis=-1)    

def _matmat(A, B, *, transpose=False, conj=False):
    if transpose:
        A = A.mT
    if conj:
        A = A.conj()
    return np.matmul(A, B)


def _transpose(A, *, conj=False):
    if not conj:
        return A.mT
    return np.conj(A).mT

def _evaluate(expr, *, locals=None, **_locals):
    """ evaluate element-wise expression """
    _locals.update(locals or {})
    return eval(expr, vars(np), _locals)


# linalg 

def linsolve(H, b, *, mu=None, maxcond=1e10):
    """ solve linear system"""
    if mu is not None:
        H = H.copy()
        inds = np.arange(H.shape[-1])
        H[..., inds, inds] += np.asanyarray(mu)[..., NAX]
    try:
        return np.linalg.solve(H, b[..., NAX])[..., 0]
    except np.linalg.LinAlgError as exc:
        pass
    valid = np.linalg.cond(H) < maxcond
    x = np.zeros_like(b)
    if np.any(valid):
        x[valid] = np.linalg.solve(H[valid], b[valid, ..., NAX])[..., 0]
    return x    



# numexpr 

try:
    import numexpr as ne

    def evaluate(expr, *, locals=None, **_locals):
        """ parallel evaluate element-wise expression """
        _locals.update(locals or {})
        return ne.evaluate(expr, local_dict=_locals)
    
except ImportError:
    evaluate = _evaluate



# numba

try:
    if not PMATH_NUMBA:
        raise ImportError()
    import numba as nb

    @nb.njit(parallel=True, cache=True)
    def matvec(A, b, transpose=False, conj=False):
        """ parallel matrix vector product """
        shape = np.broadcast_shapes(A.shape[:-2], b.shape[:-1])
        size = np.int64(np.prod(np.array(shape)))
        A = np.ascontiguousarray(A)
        b = np.ascontiguousarray(b)
        A = A.reshape(-1, A.shape[-2], A.shape[-1])
        b = b.reshape(-1, b.shape[-1])
        if transpose:
            n2, n1 = A.shape[-2:]
        else:
            n1, n2 = A.shape[-2:]
        sizeA, sizeb = A.shape[0], b.shape[0]
        res = np.empty((size, n1), dtype=type(A.flat[0] + b.flat[0]))
        for i in nb.prange(size):
            A_ = A[i % sizeA]
            b_ = b[i % sizeb]
            for j in nb.prange(n1):
                acc = 0 * res.flat[0]
                for k in nb.prange(n2):
                    if transpose:
                        Ajk = A_[k, j]
                    else:
                        Ajk = A_[j, k]
                    if conj:
                        Ajk = np.conj(Ajk)
                    acc += Ajk * b_[k]
                res[i, j] = acc
        return res.reshape(shape + (n1,))
    

    @nb.njit(parallel=True, cache=True)
    def matmat(A, B, transpose=False, conj=False):
        """ parallel matrix matrix product """
        shape = np.broadcast_shapes(A.shape[:-2], B.shape[:-2])
        size = np.int64(np.prod(np.array(shape)))
        A = np.ascontiguousarray(A)
        B = np.ascontiguousarray(B)
        A = A.reshape(-1, A.shape[-2], A.shape[-1])
        B = B.reshape(-1, B.shape[-2], B.shape[-1])
        if transpose:
            n2, n1 = A.shape[-2:]
        else:
            n1, n2 = A.shape[-2:]
        n3 = A.shape[-2]
        sizeA, sizeB = A.shape[0], B.shape[0]
        res = np.empty((size, n1, n3), dtype=type(A.flat[0] + B.flat[0]))
        for i in nb.prange(size):
            A_ = A[i % sizeA]
            B_ = B[i % sizeB]
            for j1 in nb.prange(n1):
                for j2 in nb.prange(n3):
                    acc = 0 * res.flat[0]
                    for k in nb.prange(n2):
                        if transpose:
                            Ajk = A_[k, j1]
                        else:
                            Ajk = A_[j1, k]
                        if conj:
                            Ajk = np.conj(Ajk)
                        acc += Ajk * B_[k, j2]
                    res[i, j1, j2] = acc
        return res.reshape(shape + (n1, n3))


    @nb.njit(parallel=True, cache=True)
    def vecvec(a, b, conj=False):
        """ matrix vector product (parallel) """
        shape = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        a = np.ascontiguousarray(a)
        b = np.ascontiguousarray(b)
        a = a.reshape(-1, a.shape[-1])
        b = b.reshape(-1, b.shape[-1])
        sizea = a.shape[0]
        sizeb = b.shape[0]
        size = np.int64(np.prod(np.array(shape)))
        res = np.empty(size, dtype=type(a.flat[0] + b.flat[0]))
        for i in nb.prange(size):
            a_ = a[i % sizea]
            b_ = b[i % sizeb]
            acc = 0 * res.flat[0]
            for j in nb.prange(a_.shape[0]):
                if conj:
                    acc += np.conj(a_[j]) * b_[j]
                else:
                    acc += a_[j] * b_[j]
            res[i] = acc
        return res.reshape(shape)    
    
    @nb.njit(parallel=True, cache=True)
    def transpose(A, conj=False):
        shape = A.shape[:-2]
        n1, n2 = A.shape[-2:]
        n12 = n1 * n2
        B = np.empty(shape + (n2, n1), dtype=A.dtype)
        if conj:
            for i in nb.prange(B.size // n12):
                for j in nb.prange(n1):
                    for k in nb.prange(n2):
                        B.flat[i * n12 + k * n1 + j] = np.conj(A.flat[i * n12 + j * n2 + k])
        else:
            for i in nb.prange(B.size // n12):
                for j in nb.prange(n1):
                    for k in nb.prange(n2):
                        B.flat[i * n12 + k * n1 + j] = A.flat[i * n12 + j * n2 + k]
        return B

    

    # @nb.njit(parallel=True, nogil=True)
    # def _linsolve(A, b):
    #     shape = np.broadcast_shapes(A.shape[:-2], b.shape[:-1])
    #     A = A.reshape(-1, A.shape[-2], A.shape[-1])
    #     b = b.reshape(-1, A.shape[-1])
    #     sizeA = A.shape[0]
    #     sizeb = b.shape[0]
    #     size = np.prod(np.array(shape))
    #     res = np.empty((size, A.shape[2]), dtype=A.dtype)
    #     for i in nb.prange(size):
    #         A_ = A[i % sizeA]
    #         b_ = b[i % sizeb]
    #         res[i] = np.linalg.solve(A_, b_)
    #     return res.reshape(shape + res.shape[1:])


except ImportError as exc:
    matvec = _matvec
    matmat = _matmat
    vecvec = _vecvec
    transpose = _transpose
    


if __name__ == '__main__':
    n = 1000000
    A = np.random.uniform(-1, 1, (n, 5, 5))
    b = np.random.uniform(-1, 1, (n, 5))
    c = np.random.uniform(-1, 1, (3, 1, 5))

    matvec(A, b)
    vecvec(b, c)