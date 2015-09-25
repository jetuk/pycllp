/*
Here are various routines for performing LDL decomposition and solving
a system of linear equations, Ax = b, using forward-backward substition.



*/

inline int tri_index(int i, int j, int size, int gid) {
  /* Returns the index in lower diagonal matrix stored in an array

  Separate entries are assumed for each gid.
  */
  return (i*(i + 1)/2 + j)*size + gid;
}

inline int matrix_index(int i, int j, int m, int size, int gid) {
  /* Returns the index in a dense 2D matrix stored in an array

  Separate entries are assumed for each gid.
  */
  return (i*m + j)*size + gid;
}

__kernel void ldl(int m, int n, __global float* A, __global float* L,
                  __global float* D) {
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
      L[l] = A[matrix_index(i, j, m, gsize, gid)];
      for (k=0; k<j; k++) {
        L[l] -= L[tri_index(i, k, gsize, gid)]*L[tri_index(j, k, gsize, gid)]*D[k*gsize+gid];
      }
      L[l] /= D[j*gsize+gid];
    }
    D[i*gsize+gid] = A[matrix_index(i, i, m, gsize, gid)];
    for (k=0; k<j; k++) {
      D[i*gsize+gid] -= D[k*gsize+gid]*pown(L[tri_index(i, k, gsize, gid)], 2);
    }
    L[tri_index(i, i, gsize, gid)] = 1.0;
  }
}
