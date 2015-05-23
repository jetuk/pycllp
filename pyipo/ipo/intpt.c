/*****************************************************************************

                Implementation of the Primal-Dual Interior Point Method
                              Robert J. Vanderbei
                                28 November 1994
                                                        
******************************************************************************/         
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
        
#ifdef QuadPrec
#include "Quad.h"
#define double Quad
#else
#define high(x) (x)
#endif

#include "linalg.h"
#include "ldlt.h"
#include "myalloc.h"
#include "macros.h"
/*
#define ABS(x)    ((x) > 0   ? (x) : -(x))
#define SGN(x)    ((x) > 0   ? (1.0) : (-1.0))
#define MAX(x,y)  ((x) > (y) ? (x) : (y))
#define MIN(x,y)  ((x) < (y) ? (x) : (y))
*/

#define EPS 1.0e-6
#define MAX_ITER 200

int solver(int m,int n,int nz,int *iA, int *kA, 
		double *A, double *b, double *c, double f,
		double *x, double *y, double *w, double *z)
{
    double  *dx, *dw, *dy, *dz;                          /* step directions */
    double  *rho, *sigma, normr0, norms0;	 	 /* infeasibilites */
    double  *D, *E;			                 /* diagonal matrices */
    double  gamma, delta, mu, theta, r;                  /* parameters */

    double  *At;			 /* arrays for A^T */
    int     *iAt, *kAt;

    int     i,j,iter,v=1,status=5;	

    float   primal_obj, dual_obj, normr, norms;

    /*******************************************************************
    * Allocate memory for arrays.
    *******************************************************************/

    MALLOC(    dx, n,   double );      
    MALLOC(    dw, m,   double );      
    MALLOC(    dy, m,   double );      
    MALLOC(    dz, n,   double );      
    MALLOC(   rho, m,   double );      
    MALLOC( sigma, n,   double );      
    MALLOC(     D, n,   double );      
    MALLOC(     E, m,   double );      

    MALLOC(   At,  nz,  double );
    MALLOC(  iAt,  nz,  int );
    MALLOC(  kAt, m+1,  int );

    /**************************************************************** 
    *  Verify input.                				    *
    ****************************************************************/

    if (m < 20 && n < 20) {
	int k;
	double AA[20][20];

        for (j=0; j<n; j++) for (i=0; i<m; i++) AA[i][j] = 0;
        for (j=0; j<n; j++) {
	    for (k=kA[j]; k<kA[j+1]; k++) {
	        AA[iA[k]][j] = A[k];
	    }
        }
        printf("A <= b: \n");
	for (i=0; i<m; i++) {
	    for (j=0; j<n; j++) {
		printf(" %5.1f", AA[i][j]);
	    }
	    printf("<= %5.1f \n", b[i]);
	}
	printf("\n");

        printf("c: \n");
	for (j=0; j<n; j++) printf(" %5.1f", c[j]);
	printf("\n");
    }

    /**************************************************************** 
    *  Initialization.              				    *
    ****************************************************************/

    for (j=0; j<n; j++) {
	x[j] = 1000.0;
	z[j] = 1000.0;
    }

    for (i=0; i<m; i++) {
	w[i] = 1000.0;
	y[i] = 1000.0;
    }

    atnum(m,n,kA,iA,A,kAt,iAt,At);

    delta = 0.02;
    r     = 0.9;

    normr0 = HUGE_VAL;
    norms0 = HUGE_VAL;

    /****************************************************************
    * 	Display Banner.
    ****************************************************************/

    printf ("m = %d,n = %d,nz = %d\n",m,n,nz);
    printf(
"------------------------------------------------------------------\n"
"         |           Primal          |            Dual           |\n"
"  Iter   |  Obj Value       Infeas   |  Obj Value       Infeas   |\n"
"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n"
    );
    fflush(stdout);

    /****************************************************************
    * 	Iteration.
    ****************************************************************/

    for (iter=0; iter<MAX_ITER; iter++) {

        /*************************************************************
	* STEP 1: Compute infeasibilities.
        *************************************************************/

	smx(m,n,A,kA,iA,x,rho);
	for (i=0; i<m; i++) {
	    rho[i] = b[i] - rho[i] - w[i];
	}
	normr = sqrt( dotprod(rho,rho,m) );

	smx(n,m,At,kAt,iAt,y,sigma);
	for (j=0; j<n; j++) {
	    sigma[j] = c[j] - sigma[j] + z[j];
	}
	norms = sqrt( dotprod(sigma,sigma,n) );

        /*************************************************************
	* STEP 2: Compute duality gap.
        *************************************************************/

	gamma = dotprod(z,x,n) + dotprod(y,w,m);

        /*************************************************************
	* Print statistics.
        *************************************************************/

	primal_obj = dotprod(c,x,n) + f;
	dual_obj   = dotprod(b,y,m) + f;
	printf("%8d   %14.7e  %8.1e    %14.7e  %8.1e \n", 
		iter, primal_obj, normr, dual_obj, norms);
	fflush(stdout);

        /*************************************************************
	* STEP 2.5: Check stopping rule.
        *************************************************************/

	if ( normr < EPS && norms < EPS && gamma < EPS ) {
	    status = 0;
	    break; /* OPTIMAL */
	}
	if ( normr > 10*normr0 ) {
	    status = 2;
	    break; /* PRIMAL INFEASIBLE (unreliable) */
	}
	if ( norms > 10*norms0 ) {
	    status = 4;
	    break; /* DUAL INFEASIBLE (unreliable) */
	}

        /*************************************************************
	* STEP 3: Compute central path parameter.
        *************************************************************/

	mu = delta * gamma / (n+m);

        /*************************************************************
	* STEP 4: Compute step directions.
        *************************************************************/

	for (j=0; j<n; j++) { D[j] = z[j]/x[j]; }
	for (i=0; i<m; i++) { E[i] = w[i]/y[i]; }

	ldltfac(n, m, kAt, iAt, At, E, D, kA, iA, A, v);

	for (j=0; j<n; j++) { dx[j] = sigma[j] - z[j] + mu/x[j]; }
	for (i=0; i<m; i++) { dy[i] = rho[i]   + w[i] - mu/y[i]; }

	forwardbackward(E, D, dy, dx);

	for (j=0; j<n; j++) { dz[j] = mu/x[j] - z[j] - D[j]*dx[j]; }
	for (i=0; i<m; i++) { dw[i] = mu/y[i] - w[i] - E[i]*dy[i]; }

        /*************************************************************
	* STEP 5: Ratio test to find step length.
        *************************************************************/

	theta = 0.0;
	for (j=0; j<n; j++) { 
	    if (theta < -dx[j]/x[j]) { theta = -dx[j]/x[j]; }
	    if (theta < -dz[j]/z[j]) { theta = -dz[j]/z[j]; }
	}
	for (i=0; i<m; i++) { 
	    if (theta < -dy[i]/y[i]) { theta = -dy[i]/y[i]; }
	    if (theta < -dw[i]/w[i]) { theta = -dw[i]/w[i]; }
	}
	theta = MIN( r/theta, 1.0 );

        /*************************************************************
	* STEP 6: Step to new point
        *************************************************************/

	for (j=0; j<n; j++) { 
	    x[j] = x[j] + theta*dx[j];
	    z[j] = z[j] + theta*dz[j];
	}
	for (i=0; i<m; i++) { 
	    y[i] = y[i] + theta*dy[i];
	    w[i] = w[i] + theta*dw[i];
	}


	normr0 = normr;
	norms0 = norms;
    }  	

    /****************************************************************
    * 	Free work space                                             *
    ****************************************************************/

    FREE(     w );
    FREE(     z );
    FREE(    dx );
    FREE(    dw );
    FREE(    dy );
    FREE(    dz );
    FREE(   rho );
    FREE( sigma );
    FREE(     D );
    FREE(     E );

    FREE(   At );
    FREE(  iAt );
    FREE(  kAt );

    return status;

}   /* End of solver */
