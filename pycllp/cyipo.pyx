import numpy as np
cimport numpy as np
np.import_array()

from libipo cimport solver, inv_clo as c_inv_clo, kAAt, iAAt, AAt, lp, ldltfac as c_ldltfac

cpdef hsd_solver(int m, int n,int nz,int[::1] iA, int[::1] kA,
        double[::1] A, double[::1] b, double[::1] c, double f,
        double[::1] x, double[::1] y, double[::1] w, double[::1] z,
        int verbose):
    cdef int status
    status = solver(m, n, nz, &iA[0], &kA[0], &A[0], &b[0], &c[0], f,
           &x[0], &y[0], &w[0], &z[0], verbose)

    c_inv_clo()
    return status

cpdef inv_clo():
    c_inv_clo()

cpdef ldltfac(int m, int n, int[::1] kA, int[::1] iA, double[::1] A,
      double[::1] dn, double[::1] dm, int[::1] kAt, int[::1] iAt, double[::1] At,
      int verbose):

      c_ldltfac(m, n, &kA[0], &iA[0], &A[0], &dn[0], &dm[0], &kAt[0], &iAt[0], &At[0], verbose)

cpdef getAAt():
    cdef np.npy_intp N
    N = lp.m + lp.n + 1
    pkAAt = np.PyArray_SimpleNewFromData(1, &N, np.NPY_INT, kAAt )
    N = pkAAt[lp.m + lp.n]
    piAAt = np.PyArray_SimpleNewFromData(1, &N, np.NPY_INT, iAAt )
    pAAt = np.PyArray_SimpleNewFromData(1, &N, np.NPY_FLOAT64, AAt )

    return pkAAt, piAAt, pAAt
