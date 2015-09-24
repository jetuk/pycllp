"""
This module contains implementations of Cholesky and LDL decomposition
of a positive definite matrix.

The Python variants are very slow, and a user is deferred to those implementations
in numpy.linalg if speed is important. They are used here as a prototype for
testing and implementing those same algorithms in OpenCL.
"""
import numpy as np


def cholesky(A,):
    """ Returns a Cholesky decomposition of A, L such that,
        A = LL'

    Reference,
        https://en.wikipedia.org/wiki/Cholesky_decomposition
    """
    L = np.zeros(A.shape)

    for i in range(A.shape[0]):
        for j in range(i):
            L[i, j] = A[i, j] - np.sum([L[i, k]*L[j, k] for k in range(j)])
            L[i, j] /= L[j, j]
        L[i, i] = np.sqrt(A[i, i] - np.sum([L[i, k]**2 for k in range(i)]))

    return L


def ldl(A,):
    """ Returns a LDL decomposition of A, (D, L) such that,
        A = LDL'

    Reference,
        https://en.wikipedia.org/wiki/Cholesky_decomposition
    """
    D = np.zeros(A.shape[0])
    L = np.zeros(A.shape)

    for i in range(A.shape[0]):
        for j in range(i):
            L[i, j] = A[i, j] - np.sum([L[i, k]*L[j, k]*D[k] for k in range(j)])
            L[i, j] /= D[j]
        D[i] = A[i, i] - np.sum([D[k]*L[i, k]**2 for k in range(i)])
        L[i, i] = 1.0

    return D, L


def forward_backward(L, U, b):
    """
    Solve the system LUx = b, where L and U are upper and lower diagonal
    matrices respectively using forward-backward substitution.
    """
    x = np.zeros(L.shape[0])

    # First let Ly = b, solve for y
    for i in range(L.shape[0]):
        x[i] = (b[i] - np.dot(x[:i], L[i, :i])) / L[i, i]

    # Now LUx = Ly -> Ux = y, solve for x
    for i in reversed(range(U.shape[0])):
        x[i] = (x[i] - np.dot(x[i+1:], U[i, i+1:])) / U[i, i]

    return x


def ldl_forward_backward(A, b):
    """
    Solve the system Ax = b using LDL decomposition.

    This function is a prototype for a OpenCL version of this algorithm. Instead
    of calculating the L and D prior to the forward-backward substitution the
    L matrix is calculated on the fly. During the forward substition, it is then
    reused during the backward substituion.
    """
    x = np.zeros(A.shape[0])
    D = np.zeros(A.shape[0])
    L = np.zeros(A.shape)

    # First let Ly = b, solve for y
    for i in range(A.shape[0]):
        for j in range(i):
            L[i, j] = A[i, j] - np.sum([L[i, k]*L[j, k]*D[k] for k in range(j)])
            L[i, j] /= D[j]
        D[i] = A[i, i] - np.sum([D[k]*L[i, k]**2 for k in range(i)])
        L[i, i] = 1.0
        x[i] = (b[i] - np.dot(x[:i], L[i, :i]*D[:i])) / D[i]

    # Now LUx = Ly -> Ux = y, solve for x
    for i in reversed(range(A.shape[0])):
        x[i] = (x[i] - np.dot(x[i+1:], L.T[i, i+1:])) / 1.0

    return x
