

cdef extern from "ipo/hsd.c":
    int solver(int m,int n,int nz,int *iA, int *kA,
    		double *A, double *b, double *c, double f,
    		double *x, double *y, double *w, double *z, int verbose)


cdef extern from "ipo/ldlt.h":
    void inv_clo()

cpdef hsd_solver(int m, int n,int nz,int[::1] iA, int[::1] kA,
        double[::1] A, double[::1] b, double[::1] c, double f,
        double[::1] x, double[::1] y, double[::1] w, double[::1] z,
        int verbose):
    cdef int status
    status = solver(m, n, nz, &iA[0], &kA[0], &A[0], &b[0], &c[0], f,
           &x[0], &y[0], &w[0], &z[0], verbose)

    inv_clo()
    return status
