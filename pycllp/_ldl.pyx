import numpy as np
cimport numpy as np


cdef int tri_index(int i, int j):
    """
    Returns the index in lower diagonal matrix stored in an array

    Separate entries are assumed for each gid.
    """
    return i*(i + 1)/2 + j

cdef Aij_primal_normal(int i, int j, double[:, :] A, double[:] x, double[:] z):
    """
    Return the (i, j) element of the primal normal matrix,
        A(X/Z)A`

    """
    cdef int k
    cdef double a = 0.0
    for k in range(A.shape[1]):
        a += A[i, k]*x[k]*A[j, k]/z[k]
    return a


cdef RHSi_primal_normal(int i, double[:, :] A, double[:] x, double[:] z, double[:] y, double[:] b, double[:] c, double mu):
    cdef int j
    cdef double rhs = b[i]

    for j in range(A.shape[1]):
        rhs += -A[i, j]*x[j] - A[i, j]*x[j]*(c[j] - np.dot(A.T[j, :], y) + mu/x[j])/z[j]
    return -rhs


cpdef factor_primal_normal(double[:, :] A, double[:] x, double[:] z, double[:] y, double[:] b, double[:] c, double mu,
                           double delta, double[:] L, double[:] D):
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
    cdef int m = A.shape[0]
    cdef int i, j
    cdef double beta = np.sqrt(np.amax([Aij_primal_normal(i, j, A, x, z) for i in range(m) for j in range(m)]))

    # First let Ly = b, solve for y
    for j in range(m):
        Dj = Aij_primal_normal(j, j, A, x, z) - np.sum([D[k]*L[tri_index(j, k)]**2 for k in range(j)])

        theta = 0.0
        for i in range(j+1, m):
            L[tri_index(i, j)] = Aij_primal_normal(i, j, A, x, z) - np.sum([L[tri_index(i, k)]*L[tri_index(j, k)]*D[k] for k in range(j)])
            theta = max(theta, abs(L[tri_index(i, j)]))

        # Apply the maximum constraint to the diagonal values
        # as described in Nocedal & Wright
        D[j] = max(abs(Dj), (theta/beta)**2, delta)

        for i in range(j+1, m):
            L[tri_index(i, j)] /= D[j]

        L[tri_index(j, j)] = 1.0


def forward_backward_ldl(double[:] L, double[:] D, double[:] b):
    """
    Solve the system LDL'x = b, where L and U are upper and lower diagonal
    matrices respectively using forward-backward substitution.
    """
    x = np.zeros(L.shape[0])

    # First let Ly = b, solve for y
    for i in range(D.shape[0]):
        x[i] = b[i]
        for j in range(i):
            x[i] -= x[j]*L[tri_index(i, j)]*D[j]
        x[i] /= D[i]

    # Now LUx = Ly -> Ux = y, solve for x
    for i in reversed(range(D.shape[0])):
        #x[i] = (x[i] - np.dot(x[i+1:], L.T[i, i+1:]))
        for j in range(i+1, D.shape[0]):
            x[i] -= x[j]*L[tri_index(j, i)]

    return x

cdef forward_backward_primal_normal(double[:] L, double[:] D, double[:, :] A, double[:] x, double[:] z, double[:] y,
                                   double[:] b, double[:] c, double mu, double[:] s):
    """
    Solve the system LUx = b, where L and U are upper and lower diagonal
    matrices respectively using forward-backward substitution.
    """
    cdef int i, j
    # First let LDy = b, solve for y
    for i in range(D.shape[0]):
        s[i] = RHSi_primal_normal(i, A, x, z, y, b, c, mu)
        for j in range(i):
            s[i] -= s[j]*L[tri_index(i, j)]*D[j]

        s[i] /= D[i]

    # Now LUx = Ly -> Ux = y, solve for x
    for i in reversed(range(D.shape[0])):
        #s[i] = (s[i] - np.dot(s[i+1:], L[i+1:, i])) / 1.0
        for j in range(i+1, D.shape[0]):
            s[i] -= s[j]*L[tri_index(j, i)]


cdef residual_primal_normal(double[:] dy, double[:, :] A, double[:] x, double[:] z, double[:] y, double[:] b,
                           double[:] c, double mu, double[:] r):
    cdef int m = A.shape[0]
    cdef int i

    for i in range(m):
        r[i] = RHSi_primal_normal(i, A, x, z, y, b, c, mu)
        for j in range(m):

            r[i] -= Aij_primal_normal(i, j, A, x, z)*dy[j]


def solve_primal_normal(A, x, z, y, b, c, mu, delta=1e-6, tolerance=1e-6):
    cdef int m = A.shape[0]

    cdef double[:] L = np.empty(m*(m+1)/2, dtype=np.float64)
    cdef double[:] D = np.empty(m, dtype=np.float64)
    cdef double[:] dy = np.empty(m, dtype=np.float64)
    cdef double[:] r = np.empty(m, dtype=np.float64)

    factor_primal_normal(A, x, z, y, b, c, mu, delta, L, D)
    forward_backward_primal_normal(L, D, A, x, z, y, b, c, mu, dy)
    residual_primal_normal(dy, A, x, z, y, b, c, mu, r)
    cdef int nref = 0
    while np.max(r) > tolerance and nref < 5:
        print("Refinement!", nref, np.max(r), dy)
        forward_backward_ldl(L, D, r)
        for i in range(m):
            dy[i] += r[i]
        residual_primal_normal(dy, A, x, z, y, b, c, mu, r)
        nref += 1

    return np.array(dy)