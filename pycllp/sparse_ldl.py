"""
This module contains implementations of sparse Cholesky and LDL decomposition
of a positive definite matrix.

The Python variants are very slow, and a user is deferred to those implementations
in numpy.linalg if speed is important. They are used here as a prototype for
testing and implementing those same algorithms in OpenCL.
"""
import numpy as np

from ._ldl import solve_primal_normal, factor_primal_normal

def cholesky(A, indptr, indices):
    """ Returns a sparse Cholesky decomposition of A, L such that,
        A = LL'



    Reference,
        https://en.wikipedia.org/wiki/Cholesky_decomposition
    """
    n = A.shape[0]
    L = np.zeros(len(indices), dtype=A.dtype)

    for j in range(n):
        Ljj = A[j, j]
        # L is known to be lower triangular, therefore the last indptr for a given
        # row is the diagonal element.
        # The diagonal element is not included in the L**2 summation ...
        for k in range(indptr[j], indptr[j+1]-1):
            Ljj -= L[k]**2
        Ljj = np.sqrt(Ljj)
        # ... but is used to update the value of L[j, j]
        L[indptr[j+1]-1] = Ljj

        for i in range(j+1, n):
            # First find jth column for row i
            k = indptr[i]
            while k < indptr[i+1]:
                if indices[k] == j:
                    break
                k += 1

            if k == indptr[i+1]:
                # This element L[i, j] of the decomposition does not exist, skip
                continue

            # Otherwise compute the value of the decomposition.
            Lij = A[i, j]

            ik = indptr[i]
            jk = indptr[j]
            # Again ignore diagonal element on j
            # The while condition, specfically j conditional, stops at the correct column
            while ik < indptr[i+1] and jk < (indptr[j+1]-1):
                icol = indices[ik]
                jcol = indices[jk]
                if icol == jcol:
                    Lij -= L[ik]*L[jk]
                    ik += 1
                    jk += 1
                elif icol < jcol:
                    ik += 1
                else:
                    jk += 1
            # Finally update the decomposition
            L[k] = Lij / Ljj

    return L


def modified_ldl(A, indptr, indices, delta=1e-6):
    """ Returns a modified Cholesky decomposition of A, (D, L) such that,
        A = LDL'

    Reference,
        Nocedal & Wright, 2006, Numerical Optimization
        This is algorithm 3.4 with modifications to D[j] as described
        in the text following the algorithm's definition.

    """
    n = A.shape[0]
    D = np.zeros(n, dtype=A.dtype)
    L = np.zeros(len(indices), dtype=A.dtype)

    beta = np.sqrt(np.amax(A))

    for j in range(n):
        Dj = A[j, j]
        # L is known to be lower triangular, therefore the last indptr for a given
        # row is the diagonal element.
        # The diagonal element is not included in the L**2 summation ...
        for k in range(indptr[j], indptr[j+1]-1):
            Dj -= D[indices[k]]*L[k]**2

        theta = 0.0

        for i in range(j+1, n):
            # First find jth column for row i
            k = indptr[i]
            while k < indptr[i+1]:
                if indices[k] == j:
                    break
                k += 1

            if k == indptr[i+1]:
                # This element L[i, j] of the decomposition does not exist, skip
                continue

            # Otherwise compute the value of the decomposition.
            Lij = A[i, j]

            ik = indptr[i]
            jk = indptr[j]
            # Again ignore diagonal element on j
            # The while condition, specfically j conditional, stops at the correct column
            while ik < indptr[i+1] and jk < (indptr[j+1]-1):
                icol = indices[ik]
                jcol = indices[jk]
                if icol == jcol:
                    Lij -= L[ik]*L[jk]*D[icol]
                    ik += 1
                    jk += 1
                elif icol < jcol:
                    ik += 1
                else:
                    jk += 1
            L[k] = Lij
            theta = max(theta, abs(Lij))

        # Apply the maximum constraint to the diagonal values
        # as described in Nocedal & Wright
        D[j] = max(abs(Dj), (theta/beta)**2, delta)

        for i in range(j+1, n):
            # First find jth column for row i
            k = indptr[i]
            while k < indptr[i+1]:
                if indices[k] == j:
                    break
                k += 1

            if k == indptr[i+1]:
                # This element L[i, j] of the decomposition does not exist, skip
                continue

            # Finally update the decomposition
            L[k] /= D[j]

        L[indptr[j+1]-1] = 1.0

    return D, L