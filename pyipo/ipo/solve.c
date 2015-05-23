/*********************************************************************/
/***    Copyright (c) Robert J. Vanderbei, 1994                    ***/
/***    All Rights Reserved                                        ***/
/*********************************************************************/

#include <string.h>
#include <math.h>
#include <stdlib.h>

#ifdef QuadPrec
#include "Quad.h"
#define double Quad
#else
#define high(x) (x)
#endif

#include "lp.h"
#include "ipo1.h"
#include "myalloc.h"

#define  NO_ITS_A_MIN  1
#define YES_ITS_A_MAX -1

int int_pt(int m,int n_old,int nz_old,int *ia, int *ka, 
		double *a,double *b, double *c, double f,
		double *x, double *y, double *w, double *z);

int solvelp( LP *lp )
{
	DECLAREALL

	int i, j, k, m0, status, nubs;

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

        /*---------------------------------------------------------+
        | abort if lower bound is not zero or upper bound is not    
	| infinite.                                               */

	for (j=0; j<n; j++) {
		if (l[j] != 0.0 ) {
			printf("nonzero lower bounds - abort \n"); exit(0);
		}
	}

        /*---------------------------------------------------------+
        | Convert equality constraints to a pair of inequalities  */

	MALLOC( kAt, 2*m+1, int );
	MALLOC( iAt, 2*nz, int );
	MALLOC(  At, 2*nz, double );
	REALLOC(  b, 2*m, double ); lp->b = b;

	atnum(m,n,kA,iA,A,kAt,iAt,At);

	m0 = m;
	nz = kAt[m];             /* just to make sure */
	for (i=0; i<m0; i++) {
	    if (r[i] < HUGE_VAL) {
		for (k=kAt[i]; k<kAt[i+1]; k++) {
		    iAt[nz] = iAt[k];
		     At[nz] =  At[k];
		     nz++;

		     At[ k] *= -1;
		}
		b[m] = b[i]+r[i];
		m++;
		kAt[m] = nz;

		b[i] *= -1;
	    } else {
		for (k=kAt[i]; k<kAt[i+1]; k++) {
		     At[ k] *= -1;
		}
		b[i] *= -1;
	    }
	}

	REALLOC( iA, nz,  int );    lp->iA = iA;
	REALLOC(  A, nz,  double ); lp->A = A;

	atnum(n,m,kAt,iAt,At,kA,iA,A);

	FREE( At ); FREE( kAt ); FREE( iAt );

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
        for (j=0; j<n; j++) {
                if (u[j] < HUGE_VAL) {
                        b[i] = u[j];
                        At[k] = 1.0; iAt[k] = j;  k++;

                        i++;
                        kAt[i] = k;
                }
        }
        m = i;
        nz = k;

        REALLOC( kA, n+1, int );    lp->kA = kA;
        REALLOC( iA, nz,  int );    lp->iA = iA;
        REALLOC(  A, nz,  double ); lp->A  =  A;

        atnum(n,m,kAt,iAt,At,kA,iA,A);

        /*---------------------------------------------------------+
        | allocate storage for solution vectors                   */

	CALLOC( x, n, double ); lp->x = x;
	CALLOC( y, m, double ); lp->y = y;
	CALLOC( w, m, double ); lp->w = w;
	CALLOC( z, n, double ); lp->z = z;

        /*---------------------------------------------------------+
        | negate objective if problem is a minimization           */

	if (lp->max != YES_ITS_A_MAX) {
	    for (j=0; j<n; j++) c[j] *= -1;
	    f *= -1;
	}
			
	status = int_pt(m,n,nz,iA,kA,A,b,c,f,x,y,w,z);

        /*---------------------------------------------------------+
        | restore objective and negate duals, if problem is a      |
	| minimization                                            */

	if (lp->max != YES_ITS_A_MAX) {
	    for (j=0; j<n; j++) { c[j] *= -1; }
	    for (i=0; i<m; i++) { y[i] *= -1; }
	    f *= -1;
	}
			
	lp->primal_obj = dotprod(c,x,n) + f;
	lp->dual_obj   = dotprod(b,y,m) + f;

	return status;
}
