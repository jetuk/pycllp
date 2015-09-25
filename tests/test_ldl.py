"""
Test of the LDL implementations provided in pycllp.ldl
"""

from pycllp.ldl import cholesky, ldl, forward_backward, ldl_forward_backward
import pytest
import numpy as np


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


def test_cholesky(A):
    # Perform decompositions
    np_chlsky = np.linalg.cholesky(A)
    py_chlsky = cholesky(A)
    # Check implementation of Cholesky above is the same as numpy
    np.testing.assert_allclose(np_chlsky, py_chlsky)


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

    py_ldl_x = ldl_forward_backward(A, b)
    np.testing.assert_allclose(np_x, py_ldl_x)


@pytest.mark.cl
def test_cl_ldl(AA):
    """ Test the CL implentation of LDL algorithm.

    This tests a series (cl_size) of matrices against the Python implementation.
    """
    # Convert to single float
    AA = AA.astype(np.float32)
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
    L = np.empty(cl_size*m*(m+1)/2, dtype=np.float32)
    D = np.empty(cl_size*m, dtype=np.float32)

    mf = cl.mem_flags
    A_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=AA)
    # Create and compile kernel
    prg = cl_krnl_ldl(ctx)
    L_g = cl.Buffer(ctx, mf.WRITE_ONLY, L.nbytes)
    D_g = cl.Buffer(ctx, mf.WRITE_ONLY, D.nbytes)

    prg.ldl(queue, (cl_size,), None, np.int32(m), np.int32(n), A_g, L_g, D_g)

    cl.enqueue_copy(queue, L, L_g)
    cl.enqueue_copy(queue, D, D_g)

    # Compare each matrix decomposition with the python equivalent.
    for i in range(cl_size):
        np.testing.assert_allclose(py_ldl_D[..., i], D[i::cl_size], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(py_ldl_L[..., i][np.tril_indices(m)], L[i::cl_size], rtol=1e-6, atol=1e-7)


def test_solve_primal_normal(m, n, b):
    from pycllp.ldl import solve_primal_normal
    # Random system matrix (not positive definite by itself)
    A = np.random.rand(m, n)
    x = np.random.rand(n)
    z = np.random.rand(n)
    y = np.random.rand(m)
    w = np.random.rand(m)

    np_dy = np.linalg.solve(-(np.eye(m)*w/y + (A*x/z).dot(A.T)), b)

    py_dy = solve_primal_normal(A, x, z, y, w, b)

    np.testing.assert_allclose(np_dy, py_dy)
