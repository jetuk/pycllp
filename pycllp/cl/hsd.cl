
#include <linalg.h>
#include <ldlt.h>

#define EPS 1.0e-7
#define MAX_ITER 200

__kernel void hsd(
  int m, int n, int denwin,
  __global float* c,
  __global float* b,
  __global float* x,
  __global float* z,
  __global float* y,
  __global float* w,
  __local float* diag,
  __global float* perm,
  __global int* iperm,
  __global float* A,
  __global int* iA,
  __global int* kA,
  __global float* At,
  __global int* iAt,
  __global int* kAt,
  __local float* AAt,
  __global int* iAAt,
  __global int* kAAt,
  __global float* Q,
  __global int* iQ,
  __global int* kQ,
  __local float* fwork,  // length 9n+10m
  __local int* iwork, // length 3n+3m
  __global int* status
  ) {
    int i, j;
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int wgid = get_group_id(0);
    float phi = 1.0;
    float psi = 1.0;
    float mu, normr, norms, dphi, dpsi;
    float delta, gamma, theta;
    float primal_obj, dual_obj;
    float temp1, temp2;
    int iter, _status;
    // Workgroup specific offsets
    int n0 = wgid*n;
    int m0 = wgid*m;
    int _max = 0;
    int ndep = 0;
    int consistent;
    float f = 0.001;

    /****************************************************************
    *  Local memory pointers         				    *
    ****************************************************************/

    __local float *dx = fwork; // n
    __local float *dz = dx+n;  // n
    __local float *dy = dz+n;  // m
    __local float *dw = dy+m;  // m
    __local float *rho = dw+m; // m
    __local float *sigma = rho+m; // n
    __local float *D = sigma+n; // n
    __local float *E = D+n;    // m
    __local float *fx = E+m;   // n
    __local float *fy = fx+n;  // m
    __local float *gx = fy+m;  // n
    __local float *gy = gx+n;  // m
    __local float *_fwork = gy+m; // remainder of local fwork

    __local int *mark = iwork;  // n+m
    __local int *_iwork = iwork+n+m; // remaind of local iwork

    /****************************************************************
    *  Initialization.              				    *
    ****************************************************************/

    for (j=0; j<n; j++) {
  	  x[n0+j] = 1.0;
  	  z[n0+j] = 1.0;
    }

    for (i=0; i<m; i++) {
  	  w[m0+i] = 1.0;
  	  y[m0+i] = 1.0;
    }

    _status = 5;
    for (iter=0; iter<MAX_ITER; iter++) {



      /*************************************************************
      * STEP 1: Compute mu and centering parameter delta.
      *************************************************************/

      dotprod(z,x,n,&temp1);
      dotprod(w,y,m,&temp2);
      mu = (temp1+temp2+phi*psi) / (n+m+1);

      if (iter%2 == 0) {
        delta = 0.0;
      } else {
        delta = 1.0;
      }

      /*************************************************************
      * STEP 1: Compute primal and dual objective function values.
      *************************************************************/

      dotprod(c,x,n,&primal_obj);
      dotprod(b,y,m,&dual_obj);

      /*************************************************************
      * STEP 2: Check stopping rule.
      *************************************************************/
      if ( isnan(mu) ) {
        _status = -2;
        break; // Bad things have happened
      }


      if ( mu < EPS ) {
        if ( phi > psi ) {
            _status = 0;
            break; /* OPTIMAL */
        }
        else
        if ( dual_obj < 0.0) {
            _status = 2;
            break; /* PRIMAL INFEASIBLE */
        }
        else
        if ( primal_obj > 0.0) {
            _status = 4;
            break; /* DUAL INFEASIBLE */
        }
        else
        {
          /* "Trouble in river city */
          _status = 4;
          break;
        }
      }

      /*************************************************************
      * STEP 3: Compute infeasibilities.
      *************************************************************/

      smx_gl(m,n,A,kA,iA,x,rho);
      for (i=0; i<m; i++) {
        rho[i] = rho[i] - b[m0+i]*phi + w[m0+i];
      }
      dotprod_ll(rho,rho,m,&normr);
      normr = sqrt( normr )/phi;

      for (i=0; i<m; i++) {
        rho[i] = -(1-delta)*rho[i] + w[m0+i] - delta*mu/y[m0+i];
      }

      smx_gl(n,m,At,kAt,iAt,y,sigma);
      for (j=0; j<n; j++) {
        sigma[j] = -sigma[j] + c[n0+j]*phi + z[n0+j];
      }

      dotprod_ll(sigma,sigma,n,&norms);
      norms = sqrt( norms )/phi;

      for (j=0; j<n; j++) {
        sigma[j] = -(1-delta)*sigma[j] + z[n0+j] - delta*mu/x[n0+j];
      }

      gamma = -(1-delta)*(dual_obj - primal_obj + psi) + psi - delta*mu/phi;

      /*************************************************************
      * Print statistics.
      *************************************************************/
      #if __OPENCL_C_VERSION__ >= CL_VERSION_1_2
      printf("%8d %8d %8d %8d   %14.7e  %8.1e    %14.7e  %8.1e  %8.1e \n",
        wgid, gid, lid, iter, (primal_obj/phi+f), normr,
              (dual_obj/phi+f),   norms, mu );
      #endif

      /*************************************************************
      * STEP 4: Compute step directions.
      *************************************************************/

      for (j=0; j<n; j++) { D[j] = z[n0+j]/x[n0+j]; }
      for (i=0; i<m; i++) { E[i] = w[m0+i]/y[m0+i]; }

      inv_num(n, m, _max, denwin, ndep,
        diag, perm, iperm, A, iA, kA, At, iAt, kAt,
        AAt, iAAt, kAAt, Q, iQ, kQ,
        E, D, _fwork, _iwork, mark);

      for (j=0; j<n; j++) { fx[j] = -sigma[j]; }
      for (i=0; i<m; i++) { fy[i] =  rho[i]; }

      forwardbackward(n, m, _max, ndep, diag, iperm, A, iA, kA, At, iAt, kAt,
      AAt, iAAt, kAAt, Q, iQ, kQ, mark,
      E, D, fy, fx, _fwork, consistent
      );

      for (j=0; j<n; j++) { gx[j] = -c[n0+j]; }
      for (i=0; i<m; i++) { gy[i] = -b[m0+i]; }

      forwardbackward(n, m, _max, ndep, diag, iperm, A, iA, kA, At, iAt, kAt,
      AAt, iAAt, kAAt, Q, iQ, kQ, mark,
      E, D, gy, gx, _fwork, consistent
      );

      dotprod_gl(c,fx,n,&temp1);
      dotprod_gl(b,fy,m,&temp2);
      dphi = temp1-temp2+gamma;

      dotprod_gl(c,gx,n,&temp1);
      dotprod_gl(b,gy,m,&temp2);
      dphi /= (temp1-temp2-psi/phi);

      for (j=0; j<n; j++) { dx[j] = fx[j] - gx[j]*dphi; }
      for (i=0; i<m; i++) { dy[i] = fy[i] - gy[i]*dphi; }

      for (j=0; j<n; j++) { dz[j] = delta*mu/x[n0+j] - z[n0+j] - D[j]*dx[j]; }
      for (i=0; i<m; i++) { dw[i] = delta*mu/y[m0+i] - w[m0+i] - E[i]*dy[i]; }
      dpsi = delta*mu/phi - psi - (psi/phi)*dphi;

      /*************************************************************
      * STEP 5: Compute step length.
      *************************************************************/

      if (iter%2 == 0) {
      } else {
          theta = 1.0;
      }
      theta = 0.0;
      for (j=0; j<n; j++) {
          if (theta < -dx[j]/x[n0+j]) { theta = -dx[j]/x[n0+j]; }
          if (theta < -dz[j]/z[n0+j]) { theta = -dz[j]/z[n0+j]; }
      }
      for (i=0; i<m; i++) {
          if (theta < -dy[i]/y[m0+i]) { theta = -dy[i]/y[m0+i]; }
          if (theta < -dw[i]/w[m0+i]) { theta = -dw[i]/w[m0+i]; }
      }

      if (theta < -dphi/phi) { theta = -dphi/phi; }
      if (theta < -dpsi/psi) { theta = -dpsi/psi; }
      theta = min( 0.95/theta, 1.0 );

      for (j=0; j<n; j++) {
    	    x[n0+j] = x[n0+j] + theta*dx[j];
    	    z[n0+j] = z[n0+j] + theta*dz[j];
    	}
    	for (i=0; i<m; i++) {
    	    y[m0+i] = y[m0+i] + theta*dy[i];
    	    w[m0+i] = w[m0+i] + theta*dw[i];
    	}

    	phi = phi + theta*dphi;
    	psi = psi + theta*dpsi;
    } // End of iteration



    for (j=0; j<n; j++) {
        x[n0+j] /= phi;
        z[n0+j] /= phi;
    }
    for (i=0; i<m; i++) {
        y[m0+i] /= phi;
        w[m0+i] /= phi;
    }

    status[wgid] = _status;

  }
