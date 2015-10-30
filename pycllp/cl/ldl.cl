/*
Here are various routines for performing LDL decomposition and solving
a system of linear equations, Ax = b, using forward-backward substition.



*/
#ifndef real
  #define real double
#endif

inline int tri_index(int i, int j, int size, int gid) {
  /* Returns the index in lower diagonal matrix stored in an array

  Separate entries are assumed for each gid.
  */
  return (i*(i + 1)/2 + j)*size + gid;
}

inline int tri_index_T(int i, int j, int size, int gid) {
  /* Returns the tranposed index in lower diagonal matrix stored in an array

  Separate entries are assumed for each gid.
  */
  return (j*(j + 1)/2 + i)*size + gid;
}

inline int matrix_index(int i, int j, int n, int size, int gid) {
  /* Returns the index in a dense 2D matrix stored in an array

  Separate entries are assumed for each gid.
  */
  return (i*n + j)*size + gid;
}

__kernel void ldl(int m, int n, __global real* A, __global real* L,
                  __global real* D) {
  /* Peform an LDL decomposition of A and save the results in L and D

  L is a lower triangular matrix. Entires are stored such that the lth
  entry of L is the i(i + 1)/2 + j entry in dense i, j  coordinates.
  */
  int i, j, k, l;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);

  for (i=0; i<n; i++) {
    // iterate through rows of L
    for (j=0; j<i; j++) {
      l = tri_index(i, j, gsize, gid);
      L[l] = A[matrix_index(i, j, n, gsize, gid)];
      for (k=0; k<j; k++) {
        L[l] -= L[tri_index(i, k, gsize, gid)]*L[tri_index(j, k, gsize, gid)]*D[k*gsize+gid];
      }
      L[l] /= D[j*gsize+gid];
    }
    D[i*gsize+gid] = A[matrix_index(i, i, n, gsize, gid)];
    for (k=0; k<j; k++) {
      D[i*gsize+gid] -= D[k*gsize+gid]*pown(L[tri_index(i, k, gsize, gid)], 2);
    }
    L[tri_index(i, i, gsize, gid)] = 1.0;
  }
}


real AXZAt_ij(int i, int j, int m, int n, int size, int gid, __global real* A,
            __global real* x, __global real* z) {
  /* Compute the (i, j) entry of matrix A(X/Z)A'
  */
  real a = 0.0;
  int k;

  for (k=0; k<n; k++) {
    // recall A is constant so its indexing is not specific to the work-item
    // x and z are work-item specific
    a += A[i*n+k]*x[k*size+gid]*A[j*n+k]/z[k*size+gid];
  }
  return a;
}

real AXZAt_ii(int i, int m, int n, int size, int gid, __global real* A,
            __global real* x, __global real* z) {
  /* Compute the (i, i) entry of matrix A(X/Z)A'
  */
  real a = 0.0;
  int k;

  for (k=0; k<n; k++) {
    // recall A is constant so its indexing is not specific to the work-item
    // x and z are work-item specific
    a += pown(A[i*n+k], 2)*x[k*size+gid]/z[k*size+gid];
  }
  return a;
}

real primal_normal_rhs_i(int i, int m, int n, int size, int gid, __global real* A,
  __global real* x, __global real* z, __global real* y, __global real* w,
  __global real* b,  __global real* c, real mu) {
  /* Compute the ith element of the right-hand side vector of the normal equations

    RHS = b - Ax - mu/Y - (AX/Z)*(c - A'y + mu/X)
  */
  real rhs = b[i*size+gid] - mu/y[i*size+gid];
  real Aty;
  int j, k;

  for (j=0; j<n; j++) {
    Aty = 0.0;
    for (k=0; k<m; k++) {
      Aty += A[k*n+j]*y[k*size+gid];
    }
    rhs += -A[i*n+j]*x[j*size+gid];
    rhs += -A[i*n+j]*x[j*size+gid]*(c[j*size+gid] - Aty + mu/x[j*size+gid])/z[j*size+gid];
  }
  return rhs;
}

__kernel void solve_primal_normal(int m, int n, __global real* A,
  __global real* x, __global real* z, __global real* y, __global real* w,
  __global real* b, __global real* c, real mu,
  __global real* L, __global real* D, __global real* dy) {
  /* Solve the system of normal equations in primal form,
      -(W/Y + A(X/Z)A')dy = b

  This kernel uses a single A matrix (in constant memory) to solve multiple
  systems with different x, z, y, w and b. Therefore this can be used to
  solve variations of the same linear programme with same constraint matrix,
  but different objective functions and constraint bounds.

  This kernel performs forward-backward substitution on the fly.

  The left side of the system is also computed as required. The notation in this
  context is that X, Z, Y, W and diagonal matrices of their respective vectors.

  L is a lower triangular matrix. Entires are stored such that the lth
  entry of L is the i(i + 1)/2 + j entry in dense i, j  coordinates.

  Reference,
      Vanderbei, R.J., Linear Programming, International Series in Operations
      Research & Management Science 196, DOI 10.1007/978-1-4614-7630-6_19
  */
  int i, j, k, l;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);

  for (i=0; i<m; i++) {
    // iterate through rows of L
    for (j=0; j<i; j++) {
      l = tri_index(i, j, gsize, gid);
      L[l] = -AXZAt_ij(i, j, m, n, gsize, gid, A, x, z);
      for (k=0; k<j; k++) {
        L[l] -= L[tri_index(i, k, gsize, gid)]*L[tri_index(j, k, gsize, gid)]*D[k*gsize+gid];
      }
      L[l] /= D[j*gsize+gid];
    }
    D[i*gsize+gid] = -(w[i*gsize+gid]/y[i*gsize+gid] + AXZAt_ii(i, m, n, gsize, gid, A, x, z));
    for (k=0; k<j; k++) {
      D[i*gsize+gid] -= D[k*gsize+gid]*pown(L[tri_index(i, k, gsize, gid)], 2);
    }
    L[tri_index(i, i, gsize, gid)] = 1.0;

    // Forward substitution
    dy[i*gsize+gid] = primal_normal_rhs_i(i, m, n, gsize, gid, A, x, z, y, w, b, c, mu);
    for (j=0; j<i; j++) {
      dy[i*gsize+gid] -= dy[j*gsize+gid]*L[tri_index(i, j, gsize, gid)]*D[j*gsize+gid];
    }
    dy[i*gsize+gid] /= D[i*gsize+gid];
  }

  // Backward substitution
  for (i=m-1; i>=0; i--) {
    for (j=i+1; j<m; j++) {
      dy[i*gsize+gid] -= dy[j*gsize+gid]*L[tri_index_T(i, j, gsize, gid)];
    }
  }
}
