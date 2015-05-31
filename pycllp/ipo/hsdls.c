/*****************************************************************************

         Implementation of the Homogeneous Self-Dual Long Step Method
                              Robert J. Vanderbei
				 6 August 1996
                                                        
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

#define EPS 1.0e-12
#define MAX_ITER 600

double linesearch(
    double xj,
    double zj,
    double dxj,
    double dzj,
    double beta,
    double delta,
    double mu
);

int solver(int m,int n,int nz,int *iA, int *kA, 
		double *A, double *b, double *c, double f,
		double *x, double *y, double *w, double *z)
{
    double  *dx, *dw, *dy, *dz;                          /* step directions */
    double  *fx, *fy, *gx, *gy;
    double  phi, psi, dphi, dpsi;
    double  *rho, *sigma, normr, norms;	 		 /* infeasibilites */
    double  *D, *E;			                 /* diagonal matrices */
    double  gamma, beta, delta, mu, theta;               /* parameters */

    double  *At;			 /* arrays for A^T */
    int     *iAt, *kAt;

    int     i,j,iter,v=1,status=5;	

    double  primal_obj, dual_obj;

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
    MALLOC(    fx, n,   double );      
    MALLOC(    fy, m,   double );      
    MALLOC(    gx, n,   double );      
    MALLOC(    gy, m,   double );      

    MALLOC(   At,  nz,  double );
    MALLOC(  iAt,  nz,  int );
    MALLOC(  kAt, m+1,  int );

    /**************************************************************** 
    *  Initialization.              				    *
    ****************************************************************/

    for (j=0; j<n; j++) {
	x[j] = 1.0;
	z[j] = 1.0;
    }

    for (i=0; i<m; i++) {
	w[i] = 1.0;
	y[i] = 1.0;
    }

    phi = 1.0;
    psi = 1.0;

    atnum(m,n,kA,iA,A,kAt,iAt,At);

    /****************************************************************
    * 	Display Banner.
    ****************************************************************/

    printf ("m = %d,n = %d,nz = %d\n",m,n,nz);
    printf(
"--------------------------------------------------------------------------\n"
"         |           Primal          |            Dual           |       |\n"
"  Iter   |  Obj Value       Infeas   |  Obj Value       Infeas   |  mu   |\n"
"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n"
    );
    fflush(stdout);

    /****************************************************************
    * 	Iteration.
    ****************************************************************/

    beta = 0.80;
    delta = 2*(1-beta);

    for (iter=0; iter<MAX_ITER; iter++) {

        /*************************************************************
	* STEP 1: Compute mu.
        *************************************************************/

	mu = (dotprod(z,x,n)+dotprod(w,y,m)+phi*psi) / (n+m+1);

        /*************************************************************
	* STEP 1: Compute primal and dual objective function values.
        *************************************************************/

	primal_obj = dotprod(c,x,n);
	dual_obj   = dotprod(b,y,m);

        /*************************************************************
	* STEP 2: Check stopping rule.
        *************************************************************/

	if ( mu < EPS ) {
	    if ( phi > EPS ) {
	        status = 0;
	        break; /* OPTIMAL */
	    }
	    else
	    if ( dual_obj < 0.0) {
	        status = 2;
	        break; /* PRIMAL INFEASIBLE */
	    }
	    else 
	    if ( primal_obj > 0.0) {
	        status = 4;
	        break; /* DUAL INFEASIBLE */
	    }
	    else
	    {
		status = 7;  /* NUMERICAL PROBLEM */
		break;
	    }
	}

        /*************************************************************
	* STEP 3: Compute infeasibilities.
        *************************************************************/

	smx(m,n,A,kA,iA,x,rho);
	for (i=0; i<m; i++) {
	    rho[i] = rho[i] - b[i]*phi + w[i];
	}
	normr = sqrt( dotprod(rho,rho,m) )/phi;
	for (i=0; i<m; i++) {
	    rho[i] = -(1-delta)*rho[i] + w[i] - delta*mu/y[i];
	}

	smx(n,m,At,kAt,iAt,y,sigma);
	for (j=0; j<n; j++) {
	    sigma[j] = -sigma[j] + c[j]*phi + z[j];
	}
	norms = sqrt( dotprod(sigma,sigma,n) )/phi;
	for (j=0; j<n; j++) {
	    sigma[j] = -(1-delta)*sigma[j] + z[j] - delta*mu/x[j];
	}

	gamma = -(1-delta)*(dual_obj - primal_obj + psi) + psi - delta*mu/phi;

        /*************************************************************
	* Print statistics.
        *************************************************************/

	printf("%8d   %14.7e  %8.1e    %14.7e  %8.1e  %8.1e \n", 
		iter, high(primal_obj/phi+f), high(normr), 
		      high(dual_obj/phi+f),   high(norms), high(mu) );
	fflush(stdout);

        /*************************************************************
	* STEP 4: Compute step directions.
        *************************************************************/

	for (j=0; j<n; j++) { D[j] = z[j]/x[j]; }
	for (i=0; i<m; i++) { E[i] = w[i]/y[i]; }

	ldltfac(n, m, kAt, iAt, At, E, D, kA, iA, A, v);

	for (j=0; j<n; j++) { fx[j] = -sigma[j]; }
	for (i=0; i<m; i++) { fy[i] =  rho[i]; }

	forwardbackward(E, D, fy, fx);

	for (j=0; j<n; j++) { gx[j] = -c[j]; }
	for (i=0; i<m; i++) { gy[i] = -b[i]; }

	forwardbackward(E, D, gy, gx);

	dphi = (dotprod(c,fx,n)-dotprod(b,fy,m)+gamma)/
	       (dotprod(c,gx,n)-dotprod(b,gy,m)-psi/phi);

	for (j=0; j<n; j++) { dx[j] = fx[j] - gx[j]*dphi; }
	for (i=0; i<m; i++) { dy[i] = fy[i] - gy[i]*dphi; }

	for (j=0; j<n; j++) { dz[j] = delta*mu/x[j] - z[j] - D[j]*dx[j]; }
	for (i=0; i<m; i++) { dw[i] = delta*mu/y[i] - w[i] - E[i]*dy[i]; }
	dpsi = delta*mu/phi - psi - (psi/phi)*dphi;

        /*************************************************************
	* STEP 5: Compute step length.
        *************************************************************/

	theta = 1.0;
	for (j=0; j<n; j++) { 
	    theta 
	    = 
	    MIN(theta, linesearch(x[j],z[j],dx[j],dz[j],beta,delta,mu));
	}
	for (i=0; i<m; i++) { 
	    theta 
	    = 
	    MIN(theta,linesearch(y[i],w[i],dy[i],dw[i],beta,delta,mu));
	}
	theta = MIN(theta,linesearch(phi,psi,dphi,dpsi,beta,delta,mu));
	/*
	if (theta < 4*beta/(n+m+1)) {
		printf("ratio = %10.3e \n", theta*(n+m+1)/(4*beta));
		status = 7;  
		break;
	}
	*/
	if (theta < 1.0) theta *= 0.9999;

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
	phi = phi + theta*dphi;
	psi = psi + theta*dpsi;
    }  	

    for (j=0; j<n; j++) { 
        x[j] /= phi;
        z[j] /= phi;
    }
    for (i=0; i<m; i++) { 
        y[i] /= phi;
        w[i] /= phi;
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
    FREE(    fx );
    FREE(    fy );
    FREE(    gx );
    FREE(    gy );

    FREE(   At );
    FREE(  iAt );
    FREE(  kAt );

    return status;

}   /* End of solver */


double linesearch(
    double xj,
    double zj,
    double dxj,
    double dzj,
    double beta,
    double delta,
    double mu
)
{
    double a = dxj*dzj;
    double b = zj*dxj + xj*dzj + (1-beta)*(1-delta)*mu;
    double c = xj*zj - (1-beta)*mu;
    double d = b*b - 4*a*c;

    /*
    printf("a = %10.3e b = %10.3e c = %10.3e \n", a,b,c);
    printf("root = %10.3e or %10.3e \n", (-b-sqrt(d))/(2*a), 2*c/(-b+sqrt(d)));
    */

    if (a == 0.0) {
	return -c/b;
    } else
    if (a>0) {
	if (b<0) {
	    if (d >= 0) {
	        return 2*c/(-b+sqrt(d));
	    } else {
		return HUGE_VAL;
	    }
	} else {
	    return HUGE_VAL;
	}
    } else {
	if (b<0) {
	    return 2*c/(-b+sqrt(d));
	} else {
	    return (-b-sqrt(d))/(2*a);
	}
    }
}
