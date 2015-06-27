cdef extern from "common/lp.h":
    ctypedef struct LP:
        int m
        int n

cdef extern from "ipo/hsd.c":
    int solver(int m,int n,int nz,int *iA, int *kA,
    		double *A, double *b, double *c, double f,
    		double *x, double *y, double *w, double *z, int verbose)

cdef extern from "ipo/ldlt.h":
    int *kAAt
    int *iAAt
    double *AAt
    LP *lp
    void inv_clo()

    void ldltfac(int m, int n, int *kA, int *iA, double *A, double *dn,
         double *dm, int *kAt, int *iAt, double *At, int verbose);
