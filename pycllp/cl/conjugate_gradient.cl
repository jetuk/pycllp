/*
Implementation of a basic conjugate gradient method for solving a
system of linear equations (Ax = b). The motivation for this algorithm
is to solve multiple systems with different A and b in parallel. One system
per work group, but parallel in work-items.

Matlab code from Wikipedia (https://en.wikipedia.org/wiki/Conjugate_gradient_method)

function [x] = conjgrad(A,b,x)
    r=b-A*x;
    p=r;
    rsold=r'*r;

    for i=1:1e6
        Ap=A*p;
        alpha=rsold/(p'*Ap);
        x=x+alpha*p;
        r=r-alpha*Ap;
        rsnew=r'*r;
        if sqrt(rsnew)<1e-10
              break;
        end
        p=r+rsnew/rsold*p;
        rsold=rsnew;
    end
end

*/

__kernel void conjgrad(int m, int niter, __global float *A, __global float *b, __global float *x,
  __local float *r, __local float *p) {
  /*
  This algorithm performs the conjungate gradient method. There is not
  stopping criteria, but rather a finite number of iterations are performed. The
  reason for this is due to the complexity of checking for the stop condition in
  multiple threads.

  Parameters
    m -- Size of the problem. I.e. A is (m X m), b and x are length m
    niter -- Finite number of iterations to perform.

  */

  int i,j,k, row, col, iter;
  int lid = get_local_id(0);
  int gid = get_global_id(0);
  int wgid = get_group_id(0);
  int lsize = get_local_size(0);

  int r0 = wgid*m;
  int r00 = wgid*m*m;
  int rm = wgid*(m+1);
  __private float rsold, rsnew, alpha;
  // nrows each work item has to compute.
  int nrows = m / lsize + 1;
  // setup initial r and p

  for (row = lid; row < nrows*lsize; row += lsize) {
    if (row < m) {
      printf("WGID: %d (%d) row: %d+%d\n", wgid, lid, r0, row);
      // Only compute rows within the matrix size limits
      r[r0+row] = b[r0+row];
      for (col = 0; col<m; col++) {
        r[r0+row] -= A[r00+row*m+col]*x[r0+col];
      }
      p[r0+row] = r[r0+row];
    }
  }

  // ensure all of r and p is computed.
  barrier(CLK_LOCAL_MEM_FENCE);
  // initial compute rsold in all threads
  rsold = 0.0;
  for (row = 0; row < m; row++) {
    printf("WGID: %d (%d) row: %d+%d r: %f\n", wgid, lid, r0, row, r[r0+row]);
    rsold += pow(r[r0+row], 2);
  }

  printf("WGID: %d (%d) rsold: %f\n", wgid, lid, rsold);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (iter = 0; iter<niter; iter++) {
    // Calculate alpha
    alpha = 0.0;
    for (row = 0; row < m; row++) {
      for (col = 0; col < m; col++) {
        alpha += p[r0+row]*A[r00+row*m+col]*p[r0+col];
      }
    }
    alpha = rsold/alpha;

    // Update x and r
    for (row = lid; row < nrows*lsize; row += lsize) {
      if (row < m) {
        // Only compute rows within the matrix size limits
        x[r0+row] += alpha*p[r0+row];

        for (col = 0; col<m; col++) {
          r[r0+row] -= alpha*A[r00+row*m+col]*p[r0+col];
        }
      }
    }
    // Calculate rnew
    rsnew = 0.0;
    for (row = 0; row < m; row++) {
      rsnew += pow(r[r0+row], 2);
    }
    // Update p
    for (row = lid; row < nrows*lsize; row += lsize) {
      if (row < m) {
        // Only compute rows within the matrix size limits
        p[r0+row] = r[r0+row] + rsnew/rsold*p[r0+row];
      }
    }
    rsold = rsnew;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
}
