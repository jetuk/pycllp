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
    D = np.zeros(A.shape[0], dtype=A.dtype)
    L = np.zeros(A.shape, dtype=A.dtype)

    for i in range(A.shape[0]):
        for j in range(i):
            L[i, j] = A[i, j] - np.sum([L[i, k]*L[j, k]*D[k] for k in range(j)])
            L[i, j] /= D[j]
        D[i] = A[i, i] - np.sum([D[k]*L[i, k]**2 for k in range(i)])
        L[i, i] = 1.0

    return D, L


def cl_krnl_ldl(context, ):
    """
    Returns a pyopencl.Program that can perform an LDL decomposition on a
    global dense A matrix.

    This function will compile, but not build the kernel.
    Use pyopencl.link_program() to link multiple Program objects in the
    calling code.
    """
    import pyopencl as cl
    from cl_tools import cl_program_from_file
    p = cl_program_from_file(context, 'ldl.cl')
    p.build()
    return p


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
    L matrix is calculated on the fly. During the forward substitution, it is then
    reused during the backward substitution.
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


def solve_primal_normal(A, x, z, y, w, b, c, mu):
    """
    Solve the system of normal equations in primal form,
        -(W/Y + A(X/Z)A`)dy = b

    This function is a prototype of an OpenCL version of this algorith. It uses
    the same method as ldl_forward_backward() to perform forward-backward
    substitution on the fly.

    The left side of the system is also computed as required. The notation in this
    context is that X, Z, Y, W and diagonal matrices of their respective vectors.

    Reference,
        Vanderbei, R.J., Linear Programming, International Series in Operations
        Research & Management Science 196, DOI 10.1007/978-1-4614-7630-6_19
    """
    m = A.shape[0]
    dy = np.zeros(m, dtype=A.dtype)
    D = np.zeros(m, dtype=A.dtype)
    L = np.zeros((m, m), dtype=A.dtype)

    def Aij(i, j):
        a = np.array(0.0, dtype=A.dtype)
        for k in range(A.shape[1]):
            a += A[i, k]*x[k]*A[j, k]/z[k]
        if i == j:
            a += w[i]/y[i]
        return -a

    def RHSi(i):
        rhs = np.array(b[i] - mu/y[i], dtype=A.dtype)
        for j in range(A.shape[1]):
            rhs += -A[i, j]*x[j] - A[i, j]*x[j]*(c[j] - np.dot(A.T[j, :], y) + mu/x[j])/z[j]
        return rhs

    # Forward substitution
    for i in range(m):
        for j in range(i):
            L[i, j] = Aij(i, j) - np.sum([L[i, k]*L[j, k]*D[k] for k in range(j)])
            L[i, j] /= D[j]
        D[i] = Aij(i, i) - np.sum([D[k]*L[i, k]**2 for k in range(i)])
        L[i, i] = 1.0
        dy[i] = (RHSi(i) - np.dot(dy[:i], L[i, :i]*D[:i])) / D[i]

    # Backward substitution
    for i in reversed(range(m)):
        dy[i] = dy[i] - np.dot(dy[i+1:], L.T[i, i+1:])

    return dy
