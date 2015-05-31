void ldltfac(
        int     m,      /* rows */
        int     n,      /* columns */
        int     *kA,    /* constraint matrix in three linear arrays */
        int     *iA, 
        double  *A, 
        double  *dn,    /* diagonal matrix for upper-left  corner */
        double  *dm,    /* diagonal matrix for lower-right corner */
        int     *kAt,   /* A^T in three linear arrays */
        int     *iAt, 
        double  *At, 
        int     verbose /* verbosity */
);

void forwardbackward(
        double  *Dn,    /* diagonal matrix for upper-left  corner */
        double  *Dm,    /* diagonal matrix for lower-right corner */
	double	*dx,
	double	*dy
);
