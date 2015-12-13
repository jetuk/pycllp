"""
Test of the LDL implementations provided in pycllp.ldl
"""

from pycllp.ldl import cholesky, modified_ldl, ldl, forward_backward, forward_backward_ldl, forward_backward_modified_ldl
import pytest
import numpy as np
DTYPE = np.float32

@pytest.fixture
def m():
    return 100


@pytest.fixture
def n():
    return 80


@pytest.fixture
def cl_size():
    return 32


@pytest.fixture
def A(m, n):
    """A random positive definite matrix"""
    A = np.random.rand(m, n)
    return np.dot(A, A.T) + np.eye(m)*m


@pytest.fixture
def AA(m, n, cl_size):
    """cl_size random positive definite matrices"""
    AA = np.empty((m, m, cl_size))
    for i in range(cl_size):
        AA[..., i] = A(m, n)
    return AA


@pytest.fixture
def b(m):
    return np.random.rand(m)


@pytest.fixture
def c(n):
    return np.random.rand(n)


def test_cholesky(A):
    # Perform decompositions
    np_chlsky = np.linalg.cholesky(A)
    py_chlsky = cholesky(A)

    # Check implementation of Cholesky above is the same as numpy
    np.testing.assert_allclose(np_chlsky, py_chlsky)
    # Check implementation of modified Cholesky algorithm
    D, L = modified_ldl(A)
    mod_chlsky = np.dot(L, np.eye(D.shape[0])*np.sqrt(D))
    np.testing.assert_allclose(np_chlsky, mod_chlsky)


def test_modified_ldl(m, n):
    A = np.random.rand(m, n)
    A = np.dot(A, A.T)
    # Perform decompositions
    np_chlsky = cholesky(A)
    D, L = modified_ldl(A)
    py_chlsky = np.dot(L, np.eye(m)*np.sqrt(D))
    # Check implementation of Cholesky above is the same as numpy
    assert np.any(np.isfinite(py_chlsky))


def test_ldl(A):
    # Perform decompositions
    np_chlsky = np.linalg.cholesky(A)
    py_ldl_D, py_ldl_L = ldl(A)
    # Check LDL decomposition multiples back to A
    np.testing.assert_allclose(A, (py_ldl_L*py_ldl_D).dot(py_ldl_L.T))


def test_solve_ldl(A, b):
    # Solve the system Ax = b
    np_x = np.linalg.solve(A, b)
    # Using python Cholesky decomposition ...
    py_chlsky = cholesky(A)
    py_chlsky_x = forward_backward(py_chlsky, py_chlsky.T, b)
    np.testing.assert_allclose(np_x, py_chlsky_x)

    # ... and LDL decomposition
    py_ldl_D, py_ldl_L = ldl(A)
    py_ldl_x = forward_backward(py_ldl_L*py_ldl_D, py_ldl_L.T, b)
    np.testing.assert_allclose(np_x, py_ldl_x)

    py_ldl_x2 = forward_backward_ldl(py_ldl_L, py_ldl_D, b)
    np.testing.assert_allclose(np_x, py_ldl_x2)

    py_ldl_x3 = forward_backward_modified_ldl(A, b)
    np.testing.assert_allclose(np_x, py_ldl_x3)


@pytest.mark.cl
def test_cl_ldl(AA):
    """ Test the CL implentation of LDL algorithm.

    This tests a series (cl_size) of matrices against the Python implementation.
    """
    # Convert to single float
    AA = AA.astype(DTYPE)
    # First calculate the Python based values for each matrix in AA
    py_ldl_D = np.empty((AA.shape[0], AA.shape[2]), dtype=AA.dtype)
    py_ldl_L = np.empty(AA.shape, dtype=AA.dtype)
    for i in range(AA.shape[2]):
        py_ldl_D[..., i], py_ldl_L[..., i] = ldl(AA[..., i])

    # Setup CL context
    import pyopencl as cl
    from pycllp.ldl import cl_krnl_ldl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Result arrays
    m, n, cl_size = AA.shape
    L = np.empty(cl_size*m*(m+1)/2, dtype=DTYPE)
    D = np.empty(cl_size*m, dtype=DTYPE)

    mf = cl.mem_flags
    A_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=AA)
    # Create and compile kernel
    prg = cl_krnl_ldl(ctx)
    L_g = cl.Buffer(ctx, mf.READ_WRITE, L.nbytes)
    D_g = cl.Buffer(ctx, mf.READ_WRITE, D.nbytes)

    # Test normal LDL (unmodified)
    prg.ldl(queue, (cl_size,), None, np.int32(m), np.int32(n), A_g, L_g, D_g)

    cl.enqueue_copy(queue, L, L_g)
    cl.enqueue_copy(queue, D, D_g)

    # Compare each matrix decomposition with the python equivalent.
    for i in range(cl_size):
        np.testing.assert_allclose(py_ldl_D[..., i], D[i::cl_size], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(py_ldl_L[..., i][np.tril_indices(m)], L[i::cl_size], rtol=1e-6, atol=1e-7)

    # Now test the modified algorithm ...
    beta = np.sqrt(np.amax(AA))
    prg.modified_ldl(queue, (cl_size,), None, np.int32(m), np.int32(n), A_g, L_g, D_g,
                     DTYPE(beta), DTYPE(1e-6))

    cl.enqueue_copy(queue, L, L_g)
    cl.enqueue_copy(queue, D, D_g)

    # Compare each matrix decomposition with the python equivalent.
    for i in range(cl_size):
        np.testing.assert_allclose(py_ldl_D[..., i], D[i::cl_size], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(py_ldl_L[..., i][np.tril_indices(m)], L[i::cl_size], rtol=1e-6, atol=1e-7)


def test_solve_primal_normal(m, n, b, c):
    from pycllp.ldl import solve_primal_normal
    # Random system matrix (not positive definite by itself)
    # Must add slack variables ot this system
    A = np.c_[np.random.rand(m, n), np.eye(m)]
    x = np.random.rand(m+n)
    z = np.random.rand(m+n)
    y = np.random.rand(m)
    # Extend c with slack variables
    cc = np.r_[c, np.zeros(m)]

    mu = 1.0
    # Create normal equations
    bb = b - A.dot(x) - (A*x/z).dot(cc - A.T.dot(y) + mu/x)
    AA = (A*x/z).dot(A.T)
    # Solve with numpy
    np_dy = np.linalg.solve(AA, -bb)
    # Check this implementation
    py_dy = solve_primal_normal(A, x, z, y, b, cc, mu, delta=0.0)

    np.testing.assert_allclose(np_dy, py_dy)


@pytest.mark.cl
def test_cl_solve_primal_normal_ldl(m, n, cl_size):
    """ Test the CL implentation of LDL algorithm.

    This tests a series (cl_size) of matrices against the Python implementation.
    """
    from pycllp.ldl import solve_primal_normal

    # Random system matrix (not positive definite by itself)
    A = np.random.rand(m, n).astype(dtype=DTYPE)
    x = np.random.rand(n, cl_size).astype(dtype=A.dtype)
    z = np.random.rand(n, cl_size).astype(dtype=A.dtype)
    y = np.random.rand(m, cl_size).astype(dtype=A.dtype)
    w = np.random.rand(m, cl_size).astype(dtype=A.dtype)
    b = np.random.rand(m, cl_size).astype(dtype=A.dtype)
    c = np.random.rand(n, cl_size).astype(dtype=A.dtype)
    mu = 1.0
    # First calculate the Python based values for each matrix in AA
    py_dy = np.empty((m, cl_size)).astype(dtype=A.dtype)
    for i in range(cl_size):
        py_dy[:, i] = solve_primal_normal(A, x[:, i], z[:, i], y[:, i], w[:, i], b[:, i], c[:, i], mu)

    # Setup CL context
    import pyopencl as cl
    from pycllp.ldl import cl_krnl_ldl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Work/Result arrays
    L = np.empty(cl_size*m*(m+1)/2, dtype=DTYPE)
    D = np.empty(cl_size*m, dtype=DTYPE)
    dy = np.empty(cl_size*m, dtype=DTYPE)

    mf = cl.mem_flags
    A_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    x_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
    z_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=z)
    y_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)
    w_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=w)
    b_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=b)
    c_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=c)
    # Create and compile kernel
    prg = cl_krnl_ldl(ctx)
    L_g = cl.Buffer(ctx, mf.READ_WRITE, L.nbytes)
    D_g = cl.Buffer(ctx, mf.READ_WRITE, D.nbytes)
    dy_g = cl.Buffer(ctx, mf.READ_WRITE, dy.nbytes)

    prg.solve_primal_normal(queue, (cl_size,), None, np.int32(m), np.int32(n),
                            A_g, x_g, z_g, y_g, w_g, b_g, c_g, DTYPE(mu), L_g, D_g, dy_g, DTYPE(1e-6))

    cl.enqueue_copy(queue, dy, dy_g)

    # Compare each solution vector, dy, with the python equivalent.
    for i in range(cl_size):
        np.testing.assert_allclose(py_dy[:, i], dy[i::cl_size], rtol=1e-5, atol=1e-5)
