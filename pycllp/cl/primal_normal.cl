/*
Here is an implementation of a path following interior point method.

*/
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define real double
#include "ldl.h"
#define EPS 1.0e-8f
#define MAX_ITER 200


__kernel void initialize_xzyw(int m, int n,
  __global double* x, __global double* z, __global double* y) {
  /* Initialize the arrays x, z, y and w to unity. */
  int i, j;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);

  for (i=0; i<m; i++) {
    y[i*gsize+gid] = 1.0;
  }
  for (j=0; j<n; j++) {
    x[j*gsize+gid] = 1.0;
    z[j*gsize+gid] = 1.0;
  }
}

double primal_infeasibility(int m, int n, __global real* A, __global double* x, __global double* b) {
    /* primal infeasibility,
      rho = b - Ax - w
      normr = sum(|rho\)
    */
  int i, j;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  double rho;
  double normr = 0.0;
  for (i=0; i<m; i++) {
    rho = b[i*gsize+gid];
    for (j=0; j<n; j++) {
      rho -= A[i*n+j]*x[j*gsize+gid];
    }
    normr += pown(rho, 2);
  }
  return sqrt(normr);
}

double dual_infeasibility(int m, int n, __global real* A, __global double* z, __global double* y, __global double* c) {
  /* dual infeasibility,
    sigma = c - A'y + z
    norms = sum(|sigma|)
  */
  int i, j;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  double sigma;
  double norms = 0.0;
  for (j=0; j<n; j++) {
    sigma = c[j*gsize+gid] + z[j*gsize+gid];
    for (i=0; i<m; i++) {
      sigma += -A[i*n+j]*y[i*gsize+gid];
    }
    norms += pown(sigma, 2);
  }
  return sqrt(norms);
}

__kernel void standard_primal_normal(int m, int n, __global real* A,
  __global double* x, __global double* z, __global double* y,
  __global double* dx, __global double* dz, __global double* dy,
  __global real* b, __global real* c, __global real* L, __global real* D,
  __global double* S, __global int* status, int verbose) {
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

  double normr0 = HUGE_VALF/10;
  double norms0 = HUGE_VALF/10;
  double normr, norms;
  double gamma, mu;
  double theta;
  double delta = 0.1f;
  double r = 0.9;
  double Aty, Atdy, tmp;

  for (iter=0; iter<MAX_ITER; iter++) {
    // calculate primal infeasibility,
    normr = primal_infeasibility(m, n, A, x, b);

    // calculate dual infeasibility,
    norms = dual_infeasibility(m, n, A, z, y, c);

    /* complementarity,
      gamma = z'x + y'w
    */
    gamma = 0.0;
    for (j=0; j<n; j++) {
      gamma += z[j*gsize+gid]*x[j*gsize+gid];
    }

    if (verbose > 1) {
      printf("%d/%d %2d |rho|: %8.1e  |sigma| %8.1e:  gamma: %8.1e\n", gid, gsize, iter, normr, norms, gamma);
    }

    /* Check stopping conditions
    */
    if(normr < EPS && norms < EPS && gamma < EPS) {
      stat = 0;
      break;  // OPTIMAL
    }

    if (normr > 10*normr0 && normr > EPS) {
      stat = 2;
      break;  // PRIMAL INFEASIBLE (unreliable)
    }

    if (norms > 10*norms0 && norms > EPS) {
      stat = 4;
      break;  // DUAL INFEASIBLE (unreliable)
    }

    // barrier parameter
    mu = delta * gamma / (n + m);

    // solve the primal normal equations
    solve_primal_normal(m, n, A, x, z, y, b, c, mu, L, D, S, dy, 1e-6);

    // compute other coordinating deltas and theta
    theta = 0.0;
    for (j=0; j<n; j++) {
      Aty = 0.0f;
      Atdy = 0.0f;
      for (i=0; i<m; i++) {
        Aty += A[i*n+j]*y[i*gsize+gid];
        Atdy += A[i*n+j]*dy[i*gsize+gid];
      }
      dx[j*gsize+gid] = (c[j*gsize+gid] - Aty + mu/x[j*gsize+gid] - Atdy)*x[j*gsize+gid]/z[j*gsize+gid];
      dz[j*gsize+gid] = (mu - z[j*gsize+gid]*dx[j*gsize+gid])/x[j*gsize+gid] - z[j*gsize+gid] ;

      theta = fmax(theta, (double)fmax(-dz[j*gsize+gid]/z[j*gsize+gid], -dx[j*gsize+gid]/x[j*gsize+gid]));
    }

    theta = fmin(r/theta, 1.0);

    for (i=0; i<m; i++) {
      y[i*gsize+gid] = y[i*gsize+gid] + theta*dy[i*gsize+gid];
    }

    for (j=0; j<n; j++) {
      z[j*gsize+gid] = z[j*gsize+gid] + theta*dz[j*gsize+gid];
      x[j*gsize+gid] = x[j*gsize+gid] + theta*dx[j*gsize+gid];
    }

    normr0 = normr;
    norms0 = norms;

  }

  status[gid] = stat;
}
