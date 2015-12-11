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
    n = A.shape[0]
    L = np.zeros(A.shape, dtype=A.dtype)

    for j in range(n):
        L[j, j] = np.sqrt(A[j, j] - np.sum([L[j, k]**2 for k in range(j)]))

        for i in range(j+1, n):
            L[i, j] = A[i, j] - np.sum([L[i, k]*L[j, k] for k in range(j)])
            L[i, j] /= L[j, j]

    return L


def modified_cholesky(A,):
    """ Returns a Cholesky decomposition of A, L such that,
        A = LL'

    Reference,
        https://en.wikipedia.org/wiki/Cholesky_decomposition
    """
    n = A.shape[0]
    L = np.zeros(A.shape, dtype=A.dtype)
    skipped = []
    for j in range(n):
        Ajj = A[j, j] - np.sum([L[j, k]**2 for k in range(j)])
        if Ajj < 1.0e-8:
            L[j, j] = 1.0e8
            skipped.append(j)
        else:
            L[j, j] = np.sqrt(Ajj)

        for i in range(j+1, n):
            L[i, j] = A[i, j] - np.sum([L[i, k]*L[j, k] for k in range(j)])
            L[i, j] /= L[j, j]

    return L, skipped


def modified_ldl(A, delta=1e-6):
    """ Returns a modified Cholesky decomposition of A, (D, L) such that,
        A = LDL'

    Reference,
        Nocedal & Wright, 2006, Numerical Optimization
        This is algorithm 3.4 with modifications to D[j] as described
        in the text following the algorithm's definition.

    """
    n = A.shape[0]
    D = np.zeros(n, dtype=A.dtype)
    L = np.zeros(A.shape, dtype=A.dtype)

    beta = np.sqrt(np.amax(A))

    for j in range(n):
        Dj = A[j, j] - np.sum([D[k]*L[j, k]**2 for k in range(j)])

        theta = 0.0
        for i in range(j+1, n):
            L[i, j] = A[i, j] - np.sum([L[i, k]*L[j, k]*D[k] for k in range(j)])
            theta = max(theta, abs(L[i, j]))
        # Apply the maximum constraint to the diagonal values
        # as described in Nocedal & Wright
        D[j] = max(abs(Dj), (theta/beta)**2, delta)

        for i in range(j+1, n):
            L[i, j] /= D[j]

        L[j, j] = 1.0

    return D, L


def ldl(A,):
    """ Returns a Cholesky decomposition in LDL form A, (D, L) such that,
        A = LDL'

    Reference,
        https://en.wikipedia.org/wiki/Cholesky_decomposition
    """
    n = A.shape[0]
    D = np.zeros(n, dtype=A.dtype)
    L = np.zeros(A.shape, dtype=A.dtype)

    for j in range(n):
        D[j] = A[j, j] - np.sum([D[k]*L[j, k]**2 for k in range(j)])
        for i in range(j+1, n):
            L[i, j] = A[i, j] - np.sum([L[i, k]*L[j, k]*D[k] for k in range(j)])
            L[i, j] /= D[j]

        L[j, j] = 1.0

    return D, L


def cl_krnl_ldl(context, ):
    """
    Returns a pyopencl.Program that can perform an LDL decomposition on a
    global dense A matrix.

    This function will compile, but not build the kernel.
    Use pyopencl.link_program() to link multiple Program objects in the
    calling code.
    """
    from .cl_tools import cl_program_from_file
    p = cl_program_from_file(context, 'ldl.cl')
    # TODO fix this to use compile() -- bug in PyOpenCL at the moment.
    p.build()
    return p


def cl_krnl_cholesky(context, ):
    """
    Returns a pyopencl.Program that can perform an Cholesky decomposition on a
    global dense A matrix.

    This function will compile, but not build the kernel.
    Use pyopencl.link_program() to link multiple Program objects in the
    calling code.
    """
    from .cl_tools import cl_program_from_file
    p = cl_program_from_file(context, 'cholesky.cl')
    # TODO fix this to use compile() -- bug in PyOpenCL at the moment.
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


def forward_backward_ldl(L, D, b):
    """
    Solve the system LDL'x = b, where L and U are upper and lower diagonal
    matrices respectively using forward-backward substitution.
    """
    x = np.zeros(L.shape[0])

    # First let Ly = b, solve for y
    for i in range(L.shape[0]):
        x[i] = (b[i] - np.dot(x[:i], L[i, :i]*D[:i])) / D[i]

    # Now LUx = Ly -> Ux = y, solve for x
    for i in reversed(range(D.shape[0])):
        x[i] = (x[i] - np.dot(x[i+1:], L.T[i, i+1:]))

    return x



def refined_forward_backward(A, L, U, b, tolerance=1e-10):
    """
    Solve the system LUx = b, where L and U are upper and lower diagonal
    matrices respectively using forward-backward substitution.

    """

    x = forward_backward(L, U, b)
    r = b - np.dot(A, x)

    while r.max() > tolerance:
        print("Refinement!")
        x += forward_backward(L, U, r)
        r = b - np.dot(A, x)

    return x


def solve_ldl(A, b):
    """
    Solve the system Ax = b using LDL decomposition.

    This function is a prototype for a OpenCL version of this algorithm. Instead
    of calculating the L and D prior to the forward-backward substitution the
    L matrix is calculated on the fly. During the forward substitution, it is then
    reused during the backward substitution.
    """
    n = A.shape[0]
    x = np.zeros(A.shape[0])
    D = np.zeros(A.shape[0])
    L = np.zeros(A.shape)

    for j in range(n):
        x[j] = b[j]

    # First let Ly = b, solve for y
    for j in range(n):
        D[j] = A[j, j] - np.sum([D[k]*L[j, k]**2 for k in range(j)])
        x[j] /= D[j]

        for i in range(j+1, n):
            L[i, j] = A[i, j] - np.sum([L[i, k]*L[j, k]*D[k] for k in range(j)])
            x[i] -= L[i, j]*x[j]
            L[i, j] /= D[j]

        L[j, j] = 1.0

    # Now LUx = Ly -> Ux = y, solve for x
    for i in reversed(range(n)):
        x[i] = (x[i] - np.dot(x[i+1:], L.T[i, i+1:])) / 1.0

    return x


def forward_backward_modified_ldl(A, b, delta=1e-6):
    """
    Solve the system Ax = b using LDL decomposition.

    This function is a prototype for a OpenCL version of this algorithm. Instead
    of calculating the L and D prior to the forward-backward substitution the
    L matrix is calculated on the fly. During the forward substitution, it is then
    reused during the backward substitution.
    """
    n = A.shape[0]
    x = np.zeros(A.shape[0])
    D = np.zeros(A.shape[0])
    L = np.zeros(A.shape)

    beta = np.sqrt(np.amax(A))

    for j in range(n):
        x[j] = b[j]

    # First let Ly = b, solve for y
    for j in range(n):
        Dj = A[j, j] - np.sum([D[k]*L[j, k]**2 for k in range(j)])

        theta = 0.0
        for i in range(j+1, n):
            L[i, j] = A[i, j] - np.sum([L[i, k]*L[j, k]*D[k] for k in range(j)])
            theta = max(theta, abs(L[i, j]))

        # Apply the maximum constraint to the diagonal values
        # as described in Nocedal & Wright
        D[j] = max(abs(Dj), (theta/beta)**2, delta)

        x[j] /= D[j]
        for i in range(j+1, n):
            x[i] -= L[i, j]*x[j]
            L[i, j] /= D[j]

        L[j, j] = 1.0

    # Now LUx = Ly -> Ux = y, solve for x
    for i in reversed(range(n)):
        x[i] = (x[i] - np.dot(x[i+1:], L.T[i, i+1:])) / 1.0

    return x


def Aij_primal_normal(i, j, A, x, z, y, w):
    """
    Return the (i, j) element of the primal normal matrix,
        -(W/Y + A(X/Z)A`)

    """
    a = np.array(0.0, dtype=A.dtype)
    for k in range(A.shape[1]):
        a += A[i, k]*x[k]*A[j, k]/z[k]
    if i == j:
        a += w[i]/y[i]
    return a


def RHSi_primal_normal(i, A, x, z, y, w, b, c, mu):
    rhs = np.array(b[i] - mu/y[i], dtype=A.dtype)
    for j in range(A.shape[1]):
        rhs += -A[i, j]*x[j] - A[i, j]*x[j]*(c[j] - np.dot(A.T[j, :], y) + mu/x[j])/z[j]
    return -rhs


def factor_primal_normal(A, x, z, y, w, b, c, mu, delta=1e-6):
    """
    Factor the system of normal equations in primal form,
        (W/Y + A(X/Z)A`)

    This function is a prototype of an OpenCL version of this algorithm. It uses
    the same method a modified Cholesky decompistion for the factorisation. The
    primal normal matrix is positive semi-definite.

    The left side of the system is also computed as required. The notation in this
    context is that X, Z, Y, W and diagonal matrices of their respective vectors.

    Reference,
        Vanderbei, R.J., Linear Programming, International Series in Operations
        Research & Management Science 196, DOI 10.1007/978-1-4614-7630-6_19
    """
    m = A.shape[0]
    D = np.zeros(m, dtype=A.dtype)
    L = np.zeros((m, m), dtype=A.dtype)

    beta = np.sqrt(np.amax([Aij_primal_normal(i, j, A, x, z, y, w) for i in range(m) for j in range(m)]))

    # First let Ly = b, solve for y
    for j in range(m):
        Dj = Aij_primal_normal(j, j, A, x, z, y, w) - np.sum([D[k]*L[j, k]**2 for k in range(j)])

        theta = 0.0
        for i in range(j+1, m):
            L[i, j] = Aij_primal_normal(i, j, A, x, z, y, w) - np.sum([L[i, k]*L[j, k]*D[k] for k in range(j)])
            theta = max(theta, abs(L[i, j]))

        # Apply the maximum constraint to the diagonal values
        # as described in Nocedal & Wright
        D[j] = max(abs(Dj), (theta/beta)**2, delta)

        for i in range(j+1, m):
            L[i, j] /= D[j]

        L[j, j] = 1.0

    return D, L


def forward_backward_primal_normal(L, D, A, x, z, y, w, b, c, mu):
    """
    Solve the system LUx = b, where L and U are upper and lower diagonal
    matrices respectively using forward-backward substitution.
    """
    s = np.zeros(L.shape[0])

    # First let LDy = b, solve for y
    for i in range(L.shape[0]):
        s[i] = (RHSi_primal_normal(i, A, x, z, y, w, b, c, mu) - np.dot(s[:i], L[i, :i]*D[:i])) / D[i]

    # Now LUx = Ly -> Ux = y, solve for x
    for i in reversed(range(L.shape[0])):
        s[i] = (s[i] - np.dot(s[i+1:], L[i+1:, i])) / 1.0

    return s


def residual_primal_normal(dy, A, x, z, y, w, b, c, mu):
    m = A.shape[0]
    r = np.zeros(m)

    for i in range(m):
        r[i] = RHSi_primal_normal(i, A, x, z, y, w, b, c, mu)
        for j in range(m):

            r[i] -= Aij_primal_normal(i, j, A, x, z, y, w)*dy[j]

    return r


def solve_primal_normal(A, x, z, y, w, b, c, mu, delta=1e-6, tolerance=1e-6):

    D, L = factor_primal_normal(A, x, z, y, w, b, c, mu, delta=delta)
    print(D)
    dy = forward_backward_primal_normal(L, D, A, x, z, y, w, b, c, mu)
    r = residual_primal_normal(dy, A, x, z, y, w, b, c, mu)
    nref = 0
    while r.max() > tolerance and nref < 5:
        print("Refinement!", nref, r.max(), dy)
        dy += forward_backward_ldl(L, D, r)
        r = residual_primal_normal(dy, A, x, z, y, w, b, c, mu)
        nref += 1

    return dy
