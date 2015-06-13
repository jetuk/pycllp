

cdef extern from "ipo/hsd.c":
    int solver(int m,int n,int nz,int *iA, int *kA,
    		double *A, double *b, double *c, double f,
    		double *x, double *y, double *w, double *z)


cdef extern from "ipo/ldlt.h":
    void inv_clo()

cpdef hsd_solver(int m, int n,int nz,int[:] iA, int[:] kA,
        double[:] A, double[:] b, double[:] c, double f,
        double[:] x, double[:] y, double[:] w, double[:] z):
    cdef int status
    status = solver(m, n, nz, &iA[0], &kA[0], &A[0], &b[0], &c[0], f,
           &x[0], &y[0], &w[0], &z[0])

    inv_clo()
    return status
