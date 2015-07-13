/*
OpenCL translation of the path following interior point method.
*/
#include <linalg2.h>
#include <ldlt2.h>

#define EPS 1.0e-6
#define MAX_ITER 200

__kernel void intpt(
  int m, int n, int lnz, int denwin,
  __global float* c,
  __global float* b,
  __global float* x,
  __global float* z,
  __global float* y,
  __global float* w,
  __global float* diag,
  __global float* perm,
  __global int* iperm,
  __global float* A,
  __global int* iA,
  __global int* kA,
  __global float* At,
  __global int* iAt,
  __global int* kAt,
  __global float* AAt,
  __global int* iAAt,
  __global int* kAAt,
  //__global float* Q,
  //__global int* iQ,
  //__global int* kQ,
  __global float* fwork,  // length 9n+10m
  __global int* iwork, // length 3n+3m
  __global int* status,
  int verbose
  ) {
    int i, j;
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int gsize = get_global_size(0);
    int wgid = get_group_id(0);
    float delta, r;
    float normr0, norms0;
    float normr, norms, gamma, mu, theta;
    float primal_obj, dual_obj;
    int n0 = gid;
    int m0 = gid;
    int iter, _status;
    int _max = 0;
    int ndep = 0;
    int consistent;
    float f = 0.0;

    /****************************************************************
    *  Local memory pointers         				    *
    ****************************************************************/

    __global float *dx = fwork; // n
    __global float *dz = dx+n*gsize;  // n
    __global float *dy = dz+n*gsize;  // m
    __global float *dw = dy+m*gsize;  // m
    __global float *rho = dw+m*gsize; // m
    __global float *sigma = rho+m*gsize; // n
    __global float *D = sigma+n*gsize; // n
    __global float *E = D+n*gsize;    // m
    __global float *_fwork = E+m*gsize; // remainder of local fwork

    __global int *mark = iwork;  // n+m
    __global int *_iwork = iwork+(n+m)*gsize; // remaind of local iwork

    /****************************************************************
    *  Initialization.              				    *
    ****************************************************************/

    for (j=n0; j<n*gsize; j+=gsize) {
  	  x[j] = 1.0;
  	  z[j] = 1.0;
    }

    for (i=m0; i<gsize*m; i+=gsize) {
  	  w[i] = 1.0;
  	  y[i] = 1.0;
    }

    delta = 0.02;
    r     = 0.9;

    normr0 = HUGE_VAL;
    norms0 = HUGE_VAL;

    _status = 5;
    for (iter=0; iter<MAX_ITER; iter++) {

      /*************************************************************
      * STEP 1: Compute infeasibilities.
      * STEP 2: Compute duality gap.
      *************************************************************/
      gamma = 0.0;

      smx(m,n,A,kA,iA,x,rho);

      normr = 0.0;
      for (i=m0; i<gsize*m; i+=gsize) {
          rho[i] = b[i] - rho[i]- w[i];
          normr += rho[i]*rho[i];
          gamma += y[i]*w[i];
      }
      normr = sqrt( normr );

      smx(n,m,At,kAt,iAt,y,sigma);
      norms = 0.0;
      for (j=n0; j<n*gsize; j+=gsize) {
          sigma[j] = c[j] - sigma[j] + z[j];
          norms += sigma[j]*sigma[j];
          gamma += x[j]*z[j];
      }
      norms = sqrt( norms );

      /*************************************************************
      * Print statistics.
      *************************************************************/
      #if __OPENCL_C_VERSION__ >= CL_VERSION_1_2
      if (verbose > 0) {
        primal_obj = 0.0;
        for (j=n0; j<n*gsize; j+=gsize) {
          primal_obj += c[j]*x[j] + f;
        }
        dual_obj = 0.0;
        for (i=m0; i<gsize*m; i+=gsize) {
      	  dual_obj += b[i]*y[i];
        }
      	printf("%8d %8d   %14.7e  %8.1e    %14.7e  %8.1e \n",
      		gid, iter, primal_obj, normr, dual_obj, norms);
      }
      #endif

      /*************************************************************
      * STEP 2.5: Check stopping rule.
      *************************************************************/

      if ( normr < EPS && norms < EPS && gamma < EPS ) {
    	    _status = 0;
    	    break; /* OPTIMAL */
    	}
    	if ( normr > 10*normr0 ) {
    	    _status = 2;
    	    break; /* PRIMAL INFEASIBLE (unreliable) */
    	}
    	if ( norms > 10*norms0 ) {
    	    _status = 4;
    	    break; /* DUAL INFEASIBLE (unreliable) */
    	}

      /*************************************************************
      * STEP 3: Compute central path parameter.
      *************************************************************/

      mu = delta * gamma / (n+m);

      /*************************************************************
      * STEP 4: Compute step directions.
      *************************************************************/

      for (j=n0; j<n*gsize; j+=gsize) { D[j] = z[j]/x[j]; }
    	for (i=m0; i<gsize*m; i+=gsize) { E[i] = w[i]/y[i]; }

      inv_num(n, m, lnz, _max, denwin, &ndep,
        diag, perm, iperm, A, iA, kA, At, iAt, kAt,
        AAt, iAAt, kAAt, //Q, iQ, kQ,
        E, D, _fwork, _iwork, mark);

    	for (j=n0; j<n*gsize; j+=gsize) {
        dx[j] = sigma[j] - z[j] + mu/x[j];
      }

    	for (i=m0; i<gsize*m; i+=gsize) {
        dy[i] = rho[i]   + w[i] - mu/y[i];
      }


      forwardbackward(n, m, lnz, _max, &ndep, diag, iperm, A, iA, kA, At, iAt, kAt,
      AAt, iAAt, kAAt, //Q, iQ, kQ,
      mark,
      E, D, dy, dx, _fwork, consistent, verbose
      );

    	for (j=n0; j<n*gsize; j+=gsize) {
        dz[j] = mu/x[j] - z[j] - D[j]*dx[j];
      }

    	for (i=m0; i<gsize*m; i+=gsize) {
        dw[i] = mu/y[i] - w[i] - E[i]*dy[i];
      }


      /*************************************************************
      * STEP 5: Ratio test to find step length.
      *************************************************************/

      theta = 0.0;
    	for (j=n0; j<n*gsize; j+=gsize) {
          if (theta < -dx[j]/x[j]) { theta = -dx[j]/x[j]; }
          if (theta < -dz[j]/z[j]) { theta = -dz[j]/z[j]; }
      }
      for (i=m0; i<gsize*m; i+=gsize) {
          if (theta < -dy[i]/y[i]) { theta = -dy[i]/y[i]; }
          if (theta < -dw[i]/w[i]) { theta = -dw[i]/w[i]; }
      }
      theta = fmin( r/theta, 1.0f );
      /*************************************************************
      * STEP 6: Step to new point
      *************************************************************/

    	for (j=n0; j<n*gsize; j+=gsize) {
          x[j] = x[j] + theta*dx[j];
          z[j] = z[j] + theta*dz[j];
      }
      for (i=m0; i<gsize*m; i+=gsize) {
          y[i] = y[i] + theta*dy[i];
          w[i] = w[i] + theta*dw[i];
      }

      normr0 = normr;
      norms0 = norms;
    }
  status[gid] = _status;
}
