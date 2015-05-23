/*********************************************************************/
/***    Copyright (c) Robert J. Vanderbei, 1995                    ***/
/***    All Rights Reserved                                        ***/
/*********************************************************************/

#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef QuadPrec
#include "Quad.h"
#define double Quad
#else
#define high(x) (x)
#endif

#include "lp.h"
#include "simpo1.h"
#include "myalloc.h"

#define  NO_ITS_A_MIN  1
#define YES_ITS_A_MAX -1

int simplex_method(int m,int n_old,int nz_old,int *ia, int *ka, 
		double *a,double *b, double *c, double f, double *x, double *y);

int solvelp( LP *lp )
{
	DECLAREALL

	int i, j, k, jj, m0, nubs, status;
	double *dworkm;

        /*---------------------------------------------------------+
        | On entry, the lp has the following form
	|
	|               T
	|     optimize c x
	|
	|     s.t. b <= Ax <= b+r
	|          l <=  x <= u
	|                                                         */

        /*---------------------------------------------------------+
        | transcribe information from lp                          */

        m       = lp->m;
        n       = lp->n;
        nz      = lp->nz;

        A       = lp->A;
        iA      = lp->iA;
        kA      = lp->kA;
        b       = lp->b;
        c       = lp->c;
        f       = lp->f;
        r       = lp->r;
        l       = lp->l;
        u       = lp->u;

	if (r == NULL) {
	    MALLOC(r, m, double);
	    for (i=0; i<m; i++) r[i] = 0.0;
	}
	if (l == NULL) {
	    CALLOC(l, n, double);
	}
	if (u == NULL) {
	    MALLOC(u, n, double);
	    for (j=0; j<n; j++) u[j] = HUGE_VAL;
	}

	m0 = m;

        /*---------------------------------------------------------+
        | abort if lower bound equals -Infinity                   */

	for (j=0; j<n; j++) {
		if (l[j] == -HUGE_VAL) {
			CALLOC( x, n, double );
			CALLOC( y, n, double );
			CALLOC( w, m, double );
			CALLOC( z, n, double );
			lp->x = x;
			lp->y = y;
			lp->w = w;
			lp->z = z;
			return 6;
		}
	}

        /*---------------------------------------------------------+
        | shift lower bounds to zero (x <- x-l) so that new problem 
        | has the following form
	|
	|               T     T
	|     optimize c x + c l
	|
	|     s.t. b-Al <= Ax <= b-Al+r
	|             0 <=  x <= u-l
	|                                                         */


	MALLOC( dworkm, m, double );

	for (j=0; j<n; j++) {
		if (u[j] != HUGE_VAL) u[j] -= l[j];
	}
	smx(m,n,A,kA,iA,l,dworkm);
	for (i=0; i<m; i++) {
		b[i] -= dworkm[i];
	}
	f += dotprod(c,l,n);

	FREE( dworkm );

        /*---------------------------------------------------------+
        | append surplus variables so that new problem 
        | has the following form
	|
	|                 T   
	|              |c| |x|    T
	|     optimize | | | | + c l
	|              |0| |w|
	|
	|                 |x| 
	|     s.t. |A -I| | | = b-Al
	|                 |w|
	|
	|             |0|    |x|    |u-l|
	|             | | <= | | <= |   |
	|             |0|    |w|    | r |
	|                                                         */

	REALLOC( c, n+m,   double ); lp->c  = c;
	REALLOC( u, n+m,   double ); lp->u  = u;
	REALLOC( kA, n+1+m,   int ); lp->kA = kA;
	REALLOC( iA, nz+m,    int ); lp->iA = iA;
	REALLOC(  A, nz+m, double ); lp->A  = A;

	j = n;
	k = nz;
	for (i=0; i<m; i++) {
		A[k] = -1.0;
		iA[k] = i;
		u[j] = r[i];
		c[j] = 0.0;

		j++; k++;
		kA[j] = k;
	}
	n  = j;
	nz = k;

        /*---------------------------------------------------------+
        | add upper bounds                                        */

	nubs = 0;
	for (j=0; j<n; j++) {
		if (u[j] < HUGE_VAL) nubs++;
	}

	MALLOC( kAt, m+1+2*nubs, int );   lp->kAt = kAt;
	MALLOC( iAt, nz+2*nubs, int );    lp->iAt = iAt;
	MALLOC(  At, nz+2*nubs, double ); lp->At  =  At;
	REALLOC(  b, m+nubs, double );    lp->b   =   b;
	REALLOC(  c, n+nubs, double );    lp->c   =   c;

	atnum(m,n,kA,iA,A,kAt,iAt,At);

	i = m;
	k = nz;
	jj = n;
	for (j=0; j<n; j++) {
		if (u[j] < HUGE_VAL) {
			c[jj] = 0.0;
			b[i] = -u[j]; 
			At[k] = -1.0; iAt[k] = j;  k++; 
			At[k] = -1.0; iAt[k] = jj; k++; jj++;

			i++;
			kAt[i] = k;
		}
	}
	m = i;  
	n = jj; 
	nz = k; 

	REALLOC( kA, n+1, int );    lp->kA = kA;
	REALLOC( iA, nz,  int );    lp->iA = iA;
	REALLOC(  A, nz,  double ); lp->A  =  A;

	CALLOC( x, n, double );
	CALLOC( y, n, double );
	CALLOC( w, m, double );
	CALLOC( z, n, double );
	lp->x = x;
	lp->y = y;
	lp->w = w;
	lp->z = z;

	atnum(n,m,kAt,iAt,At,kA,iA,A);

        /*---------------------------------------------------------+
        | negate constraints                                      */

	for (k=0; k<nz; k++) A[k] *= -1;
	for (i=0; i<m; i++) b[i] *= -1;

        /*---------------------------------------------------------+
        | negate objective if problem is a minimization           */

	if (lp->max == NO_ITS_A_MIN) {
	    for (j=0; j<n; j++) c[j] *= -1;
	    f *= -1;
	}
			
        /*---------------------------------------------------------+
        | if problem data is small, print it out.                 */

	if ( m<7 && n<7 ) {
	    printf("A: \n"); 
	      for (j=0; j<n; j++) {for (k=kA[j]; k<kA[j+1]; k++) {
	        printf("%5d %10.5f \n", iA[k], A[k]);
	      } printf("\n");
	    } printf("\n");

	    printf("b: \n"); 
	    for (i=0; i<m; i++) {printf("%10.5f \n", b[i]);} printf("\n");

	    printf("c: \n"); 
	    for (j=0; j<n; j++) {printf("%10.5f \n", c[j]);} printf("\n");
	}

	status = simplex_method(m,n,nz,iA,kA,A,b,c,f,x,y);

        /*---------------------------------------------------------+
        | restore objective and negate duals, if problem is a      |
	| minimization                                            */

	if (lp->max == NO_ITS_A_MIN) {
	    for (j=0; j<n; j++) { c[j] *= -1; }
	    for (j=0; j<n; j++) { y[j] *= -1; }
	    f *= -1;
	}
			
        /*---------------------------------------------------------+
        | the true dual variables are in the last m slots.         |
	| they need to be shifted to the first m slots.           */

	for (j=0; j<n; j++) z[j] = y[j];
	for (i=0; i<m; i++) {
	    y[i] = y[n-m+i];
	    w[i] = x[n-m+i];
	}

        /*---------------------------------------------------------+
        | the right-hand side has to be adjusted so that the       |
	| dual objective computation is correct when there are     |
	| finite ranges (including equality constraints).         */

	k = m;
	for (i=m0-1; i>=0; i--) {
	    if (r[i] < HUGE_VAL) {
		k--;
	        b[k] -= b[i];
	    }
	}

        /*---------------------------------------------------------+
        | record the primal and the dual objective value so        |
	| AMPL will be able to report it.                         */

	lp->primal_obj = dotprod(c,x,n) + f;
	lp->dual_obj   = dotprod(b,y,m) + f;

	return status;
}
