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

__kernel void modified_ldl(int m, int n, __global real* A, __global real* L,
                  __global real* D, real beta, real delta) {
  /* Peform a modified Cholesky decomposition of A and save the results in L and D

  The modified decomposition controls the diagonal elements should A not be
  exactly semi-definite. This is useful during LP solves where some elements in
  A can become very small (and possibly negative) due to floating point rounding.

  L is a lower triangular matrix. Entires are stored such that the lth
  entry of L is the i(i + 1)/2 + j entry in dense i, j  coordinates.

  Reference,
    Nocedal & Wright, 2006, Numerical Optimization
    This is algorithm 3.4 with modifications to D[j] as described
    in the text following the algorithm's definition.

  */
  int i, j, k;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  real Dj, theta, Lij;

  for (j=0; j<n; j++) {
    // Initial estimate of D[j]
    Dj = A[matrix_index(j, j, n, gsize, gid)];
    for (k=0; k<j; k++) {
      Dj -= D[k*gsize+gid]*pown(L[tri_index(j, k, gsize, gid)], 2);
    }

    theta = 0.0;
    // iterate through rows of L
    for (i=j+1; i<n; i++) {
      Lij = A[matrix_index(i, j, n, gsize, gid)];
      for (k=0; k<j; k++) {
        Lij -= L[tri_index(i, k, gsize, gid)]*L[tri_index(j, k, gsize, gid)]*D[k*gsize+gid];
      }
      theta = fmax(theta, fabs(Lij));
      L[tri_index(i, j, gsize, gid)] = Lij;
    }

    // Apply the maximum constraint to the diagonal values as described in Nocedal & Wright
    Dj = fmax(fabs(Dj), fmax(pown(theta/beta, 2), delta));

    for (i=j+1; i<n; i++) {
      L[tri_index(i, j, gsize, gid)] /= Dj;
    }

    D[j*gsize+gid] = Dj;
    L[tri_index(j, j, gsize, gid)] = 1.0;
  }
}


double AXZAt_ij(int i, int j, int m, int n, int size, int gid, __global real* A,
            __global double* x, __global double* z) {
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

double AXZAt_ii(int i, int m, int n, int size, int gid, __global real* A,
            __global double* x, __global double* z) {
  /* Compute the (i, i) entry of matrix A(X/Z)A'
  */
  real a = 0.0f;
  int k;

  for (k=0; k<n; k++) {
    // recall A is constant so its indexing is not specific to the work-item
    // x and z are work-item specific
    a += pown(A[i*n+k], 2)*x[k*size+gid]/z[k*size+gid];
  }
  return a;
}

double sparse_AXZAt_ij(int i, int j, int m, int n,
    __global real* Adata, __global int* Aindptr, __global int* Aindices,
    __global double* x, __global double *z) {
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  int ik, jk, ikk, jkk, icol, jcol;

  real a = 0.0;

  ik = Aindptr[i];
  ikk = Aindptr[i+1];
  jk = Aindptr[j];
  jkk = Aindptr[j+1];

  // now loop through the columns of the two rows (i & j) and perform
  // a += A[i, k]*x[k]*A[j, k]/z[k]
  while (ik < ikk && jk < jkk) {
    icol = Aindices[ik];
    jcol = Aindices[jk];
    // if columns are the same perform dot product
    if (icol == jcol) {
        a += Adata[ik]*Adata[jk]*x[icol*gsize+gid]/z[icol*gsize+gid];;
        ik += 1;
        jk += 1;
    } else if (icol < jcol) {
        ik += 1;
    } else {
        jk += 1;
    }
  }

  return a;
}

double sparse_AXZAt_ii(int i, int m, int n, __global real* Adata, __global int* Aindptr, __global int* Aindices,
                       __global double* x, __global double* z) {
  /* Compute the (i, i) entry of matrix A(X/Z)A'
  */
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  int ik, ikk, icol;

  real a = 0.0;

  ik = Aindptr[i];
  ikk = Aindptr[i+1];

  // now loop through the columns of the two rows (i & j) and perform
  // a += A[i, k]*x[k]*A[j, k]/z[k]
  while (ik < ikk) {
    icol = Aindices[ik];
    a += pown(Adata[ik], 2)*x[icol*gsize+gid]/z[icol*gsize+gid];;
    ik += 1;
  }

  return a;
}

double primal_normal_rhs_i(int i, int m, int n, int size, int gid, __global real* A,
  __global double* x, __global double* z, __global double* y,
  __global real* b,  __global real* c, real mu) {
  /* Compute the ith element of the right-hand side vector of the normal equations

    RHS = b - Ax - (AX/Z)*(c - A'y + mu/X)
  */

  real rhs = b[i*size+gid];
  real Aty, Ax;
  int j, k;

  for (j=0; j<n; j++) {
    Aty = 0.0f;
    for (k=0; k<m; k++) {
      Aty += A[k*n+j]*y[k*size+gid];
    }
    rhs += -A[i*n+j]*x[j*size+gid];
    rhs += -A[i*n+j]*x[j*size+gid]*(c[j*size+gid] - Aty + mu/x[j*size+gid])/z[j*size+gid];
  }
  return -rhs;
}

double sparse_primal_normal_rhs_i(int i, int m, int n, int size, int gid,
  __global real* Adata, __global int* Aindptr, __global int* Aindices,
  __global real* ATdata, __global int* ATindptr, __global int* ATindices,
  __global double* x, __global double* z, __global double* y,
  __global real* b,  __global real* c, real mu) {
  /* Compute the ith element of the right-hand side vector of the normal equations

    RHS = b - Ax - (AX/Z)*(c - A'y + mu/X)
  */

  real rhs = b[i*size+gid];
  real Aty, Ax;
  int j, it;
  int k, kk;
  int kt, kkt;

  k = Aindptr[i];
  kk = Aindptr[i+1];

  while (k < kk) {
    j = Aindices[k];
    Aty = 0.0;
    kt = ATindptr[j];
    kkt = ATindptr[j+1];

    while (kt < kkt) {
      it = ATindices[kt];
      Aty += ATdata[kt]*y[it*size+gid];
      kt += 1;
    }
    rhs += -Adata[k]*x[j*size+gid];
    rhs += -Adata[k]*x[j*size+gid]*(c[j*size+gid] - Aty + mu/x[j*size+gid])/z[j*size+gid];

    k += 1;
  }
  return -rhs;
}



double AXZAt_absmax(int m, int n, __global real* A,
  __global double* x, __global double* z) {
  /* Return max(|AXZA'|)
  */
  int i, j;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  double beta;
  // estimate beta for modified decomposition - see Nocedal & Wright
  beta = 0.0f;
  for (j=0; j<m; j++) {
    for (i=0; i<=j; i++) {
        beta = fmax(beta, fabs(AXZAt_ij(i, j, m, n, gsize, gid, A, x, z)));
    }
  }
  beta = sqrt(beta);
  return beta;
}

double AXZAt_diag_absmax(int m, int n, __global real* A, __global double* x, __global double* z) {
  /* Return max(|AXZA'|)
  */
  int i, j;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  double beta;
  // estimate beta for modified decomposition - see Nocedal & Wright
  beta = 0.0f;
  for (j=0; j<m; j++) {
    beta = fmax(beta, fabs(AXZAt_ii(j, m, n, gsize, gid, A, x, z)));
  }
  beta = sqrt(beta);
  return beta;
}

double sparse_AXZAt_diag_absmax(int m, int n, __global real* Adata, __global int* Aindptr, __global int* Aindices,
                                __global double* x, __global double* z) {
  /* Return max(|AXZA'|)
  */
  int i, j;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  double beta;
  // estimate beta for modified decomposition - see Nocedal & Wright
  beta = 0.0f;
  for (j=0; j<m; j++) {
    beta = fmax(beta, fabs(sparse_AXZAt_ii(j, m, n, Adata, Aindptr, Aindices, x, z)));
  }
  beta = sqrt(beta);
  return beta;
}


__kernel void factor_primal_normal(int m, int n, __global real* A,
  __global double* x, __global double* z, __global double* y,
  __global real* b, __global real* c, real mu,
  __global real* L, __global real* D, real delta) {
  /* Factor the matrix of normal equations in primal form,
      LDL' = (W/Y + A(X/Z)A')

  This kernel uses a modified Cholesky decomposition to create L and D.

  This kernel uses a single A matrix (in constant memory) to solve multiple
  systems with different x, z, y, w and b. Therefore this can be used to
  solve variations of the same linear programme with same constraint matrix,
  but different objective functions and constraint bounds.

  The notation in this context is that X, Z, Y, W and diagonal matrices of their
  respective vectors.

  L is a lower triangular matrix. Entires are stored such that the lth
  entry of L is the i(i + 1)/2 + j entry in dense i, j  coordinates.

  References,
      Vanderbei, R.J., Linear Programming, International Series in Operations
      Research & Management Science 196, DOI 10.1007/978-1-4614-7630-6_19

      Nocedal & Wright, 2006, Numerical Optimization
      This is algorithm 3.4 with modifications to D[j] as described
      in the text following the algorithm's definition.
  */
  int i, j, k;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  double Dj, Lij, theta, beta;

  beta = AXZAt_diag_absmax(m, n, A, x, z);

  for (j=0; j<m; j++) {
    // iterate through the columns of the matrix
    Dj = AXZAt_ii(j, m, n, gsize, gid, A, x, z);
    for (k=0; k<j; k++) {
      Dj -= D[k*gsize+gid]*pown(L[tri_index(j, k, gsize, gid)], 2);
    }

    theta = 0.0f;
    // iterate through rows of L
    for (i=j+1; i<m; i++) {
      Lij = AXZAt_ij(i, j, m, n, gsize, gid, A, x, z);
      for (k=0; k<j; k++) {
        Lij -= L[tri_index(i, k, gsize, gid)]*L[tri_index(j, k, gsize, gid)]*D[k*gsize+gid];
      }
      theta = fmax(theta, fabs(Lij));
      L[tri_index(i, j, gsize, gid)] = Lij;
    }

    // Apply the maximum constraint to the diagonal values as described in Nocedal & Wright
    Dj = fmax(fabs(Dj), fmax(pown(theta/beta, 2), delta));

    for (i=j+1; i<m; i++) {
      L[tri_index(i, j, gsize, gid)] /= Dj;
    }

    D[j*gsize+gid] = Dj;
    L[tri_index(j, j, gsize, gid)] = 1.0f;
  }

}


__kernel void sparse_factor_primal_normal(int m, int n,
  __global real* Adata, __global int* Aindptr, __global int* Aindices,
  __global double* x, __global double* z, __global double* y,
  __global real* b, __global real* c, real mu,
  __global real* Ldata, __constant int* Lindptr, __constant int* Lindices,
  __global real* D, real delta) {
  /* Factor the matrix of normal equations in primal form,
      LDL' = (W/Y + A(X/Z)A')

  This kernel uses a modified Cholesky decomposition to create L and D.

  This kernel uses a single A matrix (in constant memory) to solve multiple
  systems with different x, z, y, w and b. Therefore this can be used to
  solve variations of the same linear programme with same constraint matrix,
  but different objective functions and constraint bounds.

  The notation in this context is that X, Z, Y, W and diagonal matrices of their
  respective vectors.

  L is a lower triangular matrix. Entires are stored such that the lth
  entry of L is the i(i + 1)/2 + j entry in dense i, j  coordinates.

  References,
      Vanderbei, R.J., Linear Programming, International Series in Operations
      Research & Management Science 196, DOI 10.1007/978-1-4614-7630-6_19

      Nocedal & Wright, 2006, Numerical Optimization
      This is algorithm 3.4 with modifications to D[j] as described
      in the text following the algorithm's definition.
  */
  int i, j, k, kk, ik, jk, jk0, icol, jcol, ikk, jkk;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  double Dj, Lij, theta, beta;
  bool found;

  beta = sparse_AXZAt_diag_absmax(m, n, Adata, Aindptr, Aindices, x, z);
  //beta = 1.0;

  for (j=0; j<m; j++) {
    // iterate through the columns of the matrix
    Dj = sparse_AXZAt_ii(j, m, n, Adata, Aindptr, Aindices, x, z);

    for (k=Lindptr[j]; k<Lindptr[j+1]-1; k++) {
        Dj -= D[Lindices[k]*gsize+gid]*pown(Ldata[k*gsize+gid], 2);
    }

    jk0 = Lindptr[j];
    jkk = Lindptr[j+1]-1;

    theta = 0.0f;
    // iterate through rows of L
    for (i=j+1; i<m; i++) {
      // First find jth column for row i
      k = Lindptr[i];
      kk = Lindptr[i+1];
      found = false;
      while (k < kk) {
        if (Lindices[k] == j) {
          // k not refers to the jth column
          found = true;
          break;
        }
        k += 1;
      }

      if (found) {
        // only process columns that exist in this row
        Lij = sparse_AXZAt_ij(i, j, m, n, Adata, Aindptr, Aindices, x, z);
        // first indices for rows i and j
        ik = Lindptr[i];
        ikk = Lindptr[i+1];
        jk = jk0;
        // now loop through the columns of the two rows (i & j) and perform
        // Lij -= L[i, k]*L[j, k]*D[k]
        while (ik < ikk && jk < jkk) {
            icol = Lindices[ik];
            jcol = Lindices[jk];
            // if columns are the same perform dot product
            if (icol == jcol) {
                Lij -= Ldata[ik*gsize+gid]*Ldata[jk*gsize+gid]*D[icol*gsize+gid];
                ik += 1;
                jk += 1;
            } else if (icol < jcol) {
                ik += 1;
            } else {
                jk += 1;
            }
        }
        // finally update global Ldata array and theta.
        Ldata[k*gsize+gid] = Lij;
        theta = fmax(theta, fabs(Lij));
      }
    }

    // Apply the maximum constraint to the diagonal values as described in Nocedal & Wright
    Dj = fmax(fabs(Dj), fmax(pown(theta/beta, 2), delta));

    for (i=j+1; i<m; i++) {
      // First find jth column for row i
      k = Lindptr[i];
      kk = Lindptr[i+1];
      found = false;
      while (k < kk) {
        if (Lindices[k] == j) {
          // k not refers to the jth column
          found = true;
          break;
        }
        k += 1;
      }

      if (found) {
        Ldata[k*gsize+gid] /= Dj;
      }
    }

    D[j*gsize+gid] = Dj;
    Ldata[jkk*gsize+gid] = 1.0f;
  }

}


__kernel void forward_backward_primal_normal(int m, int n, __global real* A,
  __global double* x, __global double* z, __global double* y,
  __global real* b, __global real* c, real mu,
  __global real* L, __global real* D, __global double* S, __global double* dy) {
  /* Perform forward-backward substituion on the normal equations in
  primal form. This kernel assumes L and D contain the factorisation
  of the system (see factor_primal_normal).

  */
  int i, j, k;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  double Si, Sj;

  // Forward substitution
  for (i=0; i<m; i++) {
    Si = S[i*gsize+gid];
    for (j=0; j<i; j++) {
      Si -= S[j*gsize+gid]*L[tri_index(i, j, gsize, gid)]*D[j*gsize+gid];
    }
    S[i*gsize+gid] = Si/D[i*gsize+gid];
  }

  // Backward substitution
  for (j=m-1; j>=0; j--) {
    Sj = S[j*gsize+gid];
    for (i=j+1; i<m; i++) {
      Sj -= S[i*gsize+gid]*L[tri_index(i, j, gsize, gid)];
    }
    S[j*gsize+gid] = Sj;
    dy[j*gsize+gid] += Sj;
  }
}


__kernel void sparse_forward_backward_primal_normal(int m, int n,
  __global real* Ldata, __constant int* Lindptr, __constant int* Lindices,
  __constant int* LTindptr, __constant int* LTindices, __constant int* LTmap,
  __global real* D, __global double* S, __global double* dy) {
  /* Perform forward-backward substituion on the normal equations in
  primal form. This kernel assumes L and D contain the factorisation
  of the system (see factor_primal_normal).

  */
  int i, j, k;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  double Si, Sj;

  // Forward substitution
  for (i=0; i<m; i++) {
    Si = S[i*gsize+gid];
    for (k=Lindptr[i]; k<Lindptr[i+1]-1; k++) {
        j = Lindices[k];
        Si -= S[j*gsize+gid]*Ldata[k*gsize+gid]*D[j*gsize+gid];
    }
    S[i*gsize+gid] = Si/D[i*gsize+gid];
  }

  // Backward substitution
  for (j=m-1; j>=0; j--) {
    Sj = S[j*gsize+gid];
    for (k=LTindptr[j]+1; k<LTindptr[j+1]; k++) {
      i = LTindices[k];
      Sj -= S[i*gsize+gid]*Ldata[LTmap[k]*gsize+gid];
    }
    S[j*gsize+gid] = Sj;
    dy[j*gsize+gid] += Sj;
  }
}


double residual_primal_normal(int m, int n, __global real* A,
  __global double* x, __global double* z, __global double* y,
  __global real* b, __global real* c, real mu,
  __global double* dy, __global double* S) {
  /* Calculate the residual (error) between of the solved solution
  */
  int i, j;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  double maxr = 0.0;
  //double Aij;
  double r;

  for (i=0; i<m; i++) {
    r = primal_normal_rhs_i(i, m, n, gsize, gid, A, x, z, y, b, c, mu);
    for (j=0; j<m; j++) {
        r -= AXZAt_ij(i, j, m, n, gsize, gid, A, x, z)*dy[j*gsize+gid];
    }
    S[i*gsize+gid] = r;
    maxr = fmax(maxr, fabs(r));
  }
  return maxr;
}


__kernel void solve_primal_normal(int m, int n, __global real* A,
  __global double* x, __global double* z, __global double* y,
  __global real* b, __global real* c, real mu,
  __global real* L, __global real* D, __global double* S, __global double* dy,
  real delta) {
  /* Solve the system of normal equations in primal form,
      -(W/Y + A(X/Z)A')dy = b - Ax - mu/Y - (AX/Z)*(c - A'y + mu/X)

  This kernel uses a single A matrix (in constant memory) to solve multiple
  systems with different x, z, y, w and b. Therefore this can be used to
  solve variations of the same linear programme with same constraint matrix,
  but different objective functions and constraint bounds.

  This kernel performs refined forward-backward in order to control the error.

  The left side of the system is also computed as required. The notation in this
  context is that X, Z, Y, W and diagonal matrices of their respective vectors.

  L is a lower triangular matrix. Entires are stored such that the lth
  entry of L is the i(i + 1)/2 + j entry in dense i, j  coordinates.

  Reference,
      Vanderbei, R.J., Linear Programming, International Series in Operations
      Research & Management Science 196, DOI 10.1007/978-1-4614-7630-6_19
  */
  int i, nref;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  double maxr;
  // Perform factorisation
  factor_primal_normal(m, n, A, x, z, y, b, c, mu, L, D, delta);
  // Initialise S to the RHS
  for (i=0; i<m; i++) {
    dy[i*gsize+gid] = 0.0;
    S[i*gsize+gid] = primal_normal_rhs_i(i, m, n, gsize, gid, A, x, z, y, b, c, mu);
  }
  // Solve system
  forward_backward_primal_normal(m, n, A, x, z, y, b, c, mu, L, D, S, dy);

  // Update S to contain the residual
  maxr = residual_primal_normal(m, n, A, x, z, y, b, c, mu, dy, S);

  nref = 0;
  while (maxr > 1e-8 && nref < 5) {
    // Solve system
    forward_backward_primal_normal(m, n, A, x, z, y, b, c, mu, L, D, S, dy);

    // Update S to contain the residual
    maxr = residual_primal_normal(m, n, A, x, z, y, b, c, mu, dy, S);
    nref += 1;
  }
}


__kernel void sparse_solve_primal_normal(int m, int n,
  __global real* Adata, __global int* Aindptr, __global int* Aindices,
  __global real* ATdata, __global int* ATindptr, __global int* ATindices,
  __global double* x, __global double* z, __global double* y,
  __global real* b, __global real* c, real mu,
  __global real* Ldata, __constant int* Lindptr, __constant int* Lindices,
  __constant int* LTindptr, __constant int* LTindices, __constant int* LTmap,
  __global real* D, __global double* S, __global double* dy,
  real delta) {
  /* Solve the system of normal equations in primal form,
      -(W/Y + A(X/Z)A')dy = b - Ax - mu/Y - (AX/Z)*(c - A'y + mu/X)

  This kernel uses a single A matrix (in constant memory) to solve multiple
  systems with different x, z, y, w and b. Therefore this can be used to
  solve variations of the same linear programme with same constraint matrix,
  but different objective functions and constraint bounds.

  This kernel performs refined forward-backward in order to control the error.

  The left side of the system is also computed as required. The notation in this
  context is that X, Z, Y, W and diagonal matrices of their respective vectors.

  L is a lower triangular matrix. Entires are stored such that the lth
  entry of L is the i(i + 1)/2 + j entry in dense i, j  coordinates.

  Reference,
      Vanderbei, R.J., Linear Programming, International Series in Operations
      Research & Management Science 196, DOI 10.1007/978-1-4614-7630-6_19
  */
  int i, nref;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  double maxr;
  // Perform factorisation
  sparse_factor_primal_normal(m, n, Adata, Aindptr, Aindices, x, z, y, b, c, mu, Ldata, Lindptr, Lindices, D, delta);
  // Initialise S to the RHS
  for (i=0; i<m; i++) {
    dy[i*gsize+gid] = 0.0;
    S[i*gsize+gid] = sparse_primal_normal_rhs_i(i, m, n, gsize, gid, Adata, Aindptr, Aindices, ATdata, ATindptr, ATindices, x, z, y, b, c, mu);
  }
  // Solve system
  sparse_forward_backward_primal_normal(m, n, Ldata, Lindptr, Lindices, LTindptr, LTindices, LTmap, D, S, dy);
/*
  // Update S to contain the residual
  maxr = residual_primal_normal(m, n, A, x, z, y, b, c, mu, dy, S);

  nref = 0;
  while (maxr > 1e-8 && nref < 5) {
    // Solve system
    sparse_forward_backward_primal_normal(m, n, Ldata, Lindptr, Lindices, LTindptr, LTindices, LTmap, D, S, dy);

    // Update S to contain the residual
    maxr = residual_primal_normal(m, n, A, x, z, y, b, c, mu, dy, S);
    nref += 1;
  }
*/
}