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
#include "lp1.h"
#include "myalloc.h"

#define  NO_ITS_A_MIN  1
#define YES_ITS_A_MAX -1

int solver(int m,int n_old,int nz_old,int *ia, int *ka, 
		double *a,double *b, double *c, double f,
		double *x, double *y, double *w, double *z);

int solvelp( LP *lp )
{
	DECLAREALL

	int i, j, k, m0, nubs, status;
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

	printf("m = %d,n = %d,nz = %d \n", m, n, nz);

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

        /*---------------------------------------------------------+
        | abort if lower bound equals -Infinity                   */

	for (j=0; j<n; j++) {
		if (l[j] == -HUGE_VAL) {
			CALLOC( x, n, double ); lp->x = x;
			CALLOC( y, m, double ); lp->y = y;
			CALLOC( w, m, double ); lp->w = w;
			CALLOC( z, n, double ); lp->z = z;
			return 3;
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
        | Convert equality constraints to a pair of inequalities  */

	MALLOC( kAt, 2*m+1, int );   lp->kAt = kAt;
	MALLOC( iAt, 2*nz, int );    lp->iAt = iAt;
	MALLOC(  At, 2*nz, double ); lp->At = At;
	REALLOC(  b, 2*m, double );  lp->b = b;

	atnum(m,n,kA,iA,A,kAt,iAt,At);

	/* 01/26/01 BUG: ampl interface is giving r[i] = Inf always */
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

        /*---------------------------------------------------------+
        | add upper bounds                                        */

        nubs = 0;
        for (j=0; j<n; j++) {
                if (u[j] < HUGE_VAL) nubs++;
        }

        REALLOC( kAt, m+1+2*nubs, int );   lp->kAt = kAt;
        REALLOC( iAt, nz+2*nubs, int );    lp->iAt = iAt;
        REALLOC(  At, nz+2*nubs, double ); lp->At  =  At;
        REALLOC(   b, m+nubs, double );    lp->b   =   b;

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

        /*---------------------------------------------------------+
        | make extra room so that solver can add slack variables   |
	| if it wants.                                            */

        REALLOC(  kA, n+m+1, int );    lp->kA  = kA;
        REALLOC(  iA, nz+m,  int );    lp->iA  = iA;
        REALLOC(   A, nz+m,  double ); lp->A   =  A;
        REALLOC( kAt, n+m+1, int );    lp->kAt = kAt;
        REALLOC( iAt, nz+m,  int );    lp->iAt = iAt;
        REALLOC(  At, nz+m,  double ); lp->At  =  At;
        REALLOC(   b,  n+m,  double ); lp->b   =  b;
        REALLOC(   c,  n+m,  double ); lp->c   =  c;

        atnum(n,m,kAt,iAt,At,kA,iA,A);

        /*---------------------------------------------------------+
        | allocate storage for solution vectors                   */

	CALLOC( x, n+m, double ); lp->x = x;
	CALLOC( y, n+m, double ); lp->y = y;
	CALLOC( w,   m, double ); lp->w = w;
	CALLOC( z,   n, double ); lp->z = z;

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

        /*---------------------------------------------------------+
        | When the solver is called, the lp has the following form: 
	|
	|               T     T
	|     maximize c x + c l
	|
	|     s.t. -Ax <= -b
	|           Ax <=  b+r-l
	|            x <=  u-l
	|
	|            x >=  0
	|                                                         */

	status = solver(m,n,nz,iA,kA,A,b,c,f,x,y,w,z);

        /*---------------------------------------------------------+
	| shift by lower bounds                                   */

	for (j=0; j<n; j++) {x[j] += l[j];}

        /*---------------------------------------------------------+
        | restore objective and negate duals, if problem is a      |
	| minimization                                            */

	if (lp->max == NO_ITS_A_MIN) {
	    for (j=0; j<n; j++) { c[j] *= -1; }
	    for (i=0; i<m; i++) { y[i] *= -1; }
	    f *= -1;
	}
			
	lp->primal_obj = dotprod(c,x,n) + f;
	lp->dual_obj   = dotprod(b,y,m) + f;

	return status;
}
