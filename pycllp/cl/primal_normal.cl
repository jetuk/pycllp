/*
Here is an implementation of a path following interior point method.

*/
#define real double
#include <ldl.h>
#define EPS 1.0e-6
#define MAX_ITER 200



__kernel void initialize_xzyw(int m, int n,
  __global real* x, __global real* z, __global real* y, __global real* w) {
  /* Initialize the arrays x, z, y and w to unity. */
  int i, j;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);

  for (i=0; i<m; i++) {
    y[i*gsize+gid] = 1.0;
    w[i*gsize+gid] = 1.0;
  }
  for (j=0; j<n; j++) {
    x[j*gsize+gid] = 1.0;
    z[j*gsize+gid] = 1.0;
  }
}

__kernel void standard_primal_normal(int m, int n, __global real* A,
  __global real* x, __global real* z, __global real* y, __global real* w,
  __global real* dx, __global real* dz, __global real* dy, __global real* dw,
  __global real* b, __global real* c, __global real* L, __global real* D,
  __global int* status, int verbose) {
  /* Solve a set of linear programmes in standard form using the primal
  form of the normal equations.

  This method uses a shared constrain matrix, A, for all work-items. The
  objective function, c, and constraint bounds, b, however are varied
  per work-item.

  Usage notes:
    - The the central path coordinate arrays x, z, y and w are not
    initialized as part of this routine. They should be initialized prior to
    first calling this routine. For second or subsequent calls solutions
    will begin from the position the previous call ended in. This is a useful
    efficiency when executed repeatidly with slowly varying LPs for which the
    optimal solution is expected only differ slightly.

  */
  int i, j, iter;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  int stat = 5;

  real normr0 = HUGE_VALF;
  real norms0 = HUGE_VALF;
  real normr, norms, rho, sigma;
  real gamma, mu;
  real theta;
  real delta = 0.02;
  real r = 0.9;


  for (iter=0; iter<MAX_ITER; iter++) {
    /* primal infeasibility,
      rho = b - Ax - w
      normr = sum(|rho\)
    We don't store the entire primal infeasibility vector, but calculate
    the vector's magnitude for the stopping rule.
    */
    normr = 0.0;
    for (i=0; i<m; i++) {
      rho = b[i*gsize+gid] - w[i*gsize+gid];
      for (j=0; j<n; j++) {
        rho += -A[i*n+j]*x[j*gsize+gid];
      }
      normr += fabs(rho);
    }

    /* dual infeasibility,
      sigma = c - A'y + z
      norms = sum(|sigma|)
    Again the whole vector is not stored.
    */
    norms = 0.0;
    for (j=0; j<n; j++) {
      sigma = c[j*gsize+gid] + z[j*gsize+gid];
      for (i=0; i<m; i++) {
        sigma += -A[i*n+j]*y[i*gsize+gid];
      }
      norms += fabs(sigma);
    }

    /* complementarity,
      gamma = z'x + y'w
    */
    gamma = 0.0;
    for (i=0; i<m; i++) {
      gamma += w[i*gsize+gid]*y[i*gsize+gid];
    }
    for (j=0; j<n; j++) {
      gamma += z[j*gsize+gid]*x[j*gsize+gid];
    }

    if (verbose > 1) {
      printf("%5d %2d |rho|: %8.1e  |sigma| %8.1e:  gamma: %8.1e\n", gid, iter, normr, norms, gamma);
    }

    /* Check stopping conditions
    */
    if(normr < EPS && norms < EPS && gamma < EPS) {
      stat = 0;
      break;  // OPTIMAL
    }

    if (normr > 10*normr0 || isnan(normr)) {
      stat = 2;
      break;  // PRIMAL INFEASIBLE (unreliable)
    }

    if (norms > 10*norms0 || isnan(norms)) {
      stat = 4;
      break;  // DUAL INFEASIBLE (unreliable)
    }

    // barrier parameter
    mu = delta * gamma / (n + m);

    // solve the primal normal equations
    solve_primal_normal(m, n, A, x, z, y, w, b, c, mu, L, D, dy);
    theta = 0.0;

    // compute other coordinating deltas and theta
    //printf("  dy      dw\n");
    for (i=0; i<m; i++) {
      dw[i*gsize+gid] = (mu - y[i*gsize+gid]*w[i*gsize+gid] - w[i*gsize+gid]*dy[i*gsize+gid])/y[i*gsize+gid];

      theta = max(theta, max(-dw[i*gsize+gid]/w[i*gsize+gid], -dy[i*gsize+gid]/y[i*gsize+gid]));
      //printf("%8.1e %8.1e %8.1e %8.1e %8.1e %8.1e\n", y[i*gsize+gid], w[i*gsize+gid], dy[i*gsize+gid], dw[i*gsize+gid], -dw[i*gsize+gid]/w[i*gsize+gid], -dy[i*gsize+gid]/y[i*gsize+gid]);
    }

    //printf("  dx      dz\n");
    for (j=0; j<n; j++) {
      dx[j*gsize+gid] = (c[j*gsize+gid] + mu/x[j*gsize+gid]);
      for (i=0; i<m; i++) {
        dx[j*gsize+gid] += -A[i*n+j]*(y[i*gsize+gid] + dy[i*gsize+gid]);
      }
      //printf("%8.1e ", dx[j*gsize+gid]);
      dx[j*gsize+gid] *= x[j*gsize+gid]/z[j*gsize+gid];
      dz[j*gsize+gid] = (mu - x[j*gsize+gid]*z[j*gsize+gid] - z[j*gsize+gid]*dx[j*gsize+gid])/x[j*gsize+gid];

      theta = max(theta, max(-dz[j*gsize+gid]/z[j*gsize+gid], -dx[j*gsize+gid]/x[j*gsize+gid]));
      //printf("%d %8.1e %8.1e %8.1e %8.1e %8.1e %8.1e\n", gid, x[j*gsize+gid], z[j*gsize+gid], dx[j*gsize+gid], dz[j*gsize+gid], -dz[j*gsize+gid]/z[j*gsize+gid], -dx[j*gsize+gid]/x[j*gsize+gid]); //, x[j*gsize+gid]/z[j*gsize+gid]);
    }
    //printf("theta: %8.1e\n", theta);
    theta = min(r/theta, 1.0);

    //printf("theta: %8.1e\n", theta);

    for (i=0; i<m; i++) {
      w[i*gsize+gid] += theta*dw[i*gsize+gid];
      y[i*gsize+gid] += theta*dy[i*gsize+gid];
    }

    for (j=0; j<n; j++) {
      z[j*gsize+gid] += theta*dz[j*gsize+gid];
      x[j*gsize+gid] += theta*dx[j*gsize+gid];
    }

    normr0 = normr;
    norms0 = norms;

  }

  status[gid] = stat;
}
