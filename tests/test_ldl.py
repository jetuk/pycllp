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
def A(m, n):
    """A random positive definite matrix"""
    A = np.random.rand(m, n)
    return np.dot(A, A.T) + np.eye(m)*m


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
def test_cl_ldl(A):
    # Convert to single float
    A = A.astype(np.float32)
    # First calculate the Pyhton based values
    py_ldl_D, py_ldl_L = ldl(A)

    import pyopencl as cl
    from pycllp.ldl import cl_krnl_ldl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Result arrays
    m = A.shape[0]
    L = np.empty(m*(m+1)/2, dtype=np.float32)
    D = np.empty(m, dtype=np.float32)

    mf = cl.mem_flags
    A_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    # Create and compile kernel
    prg = cl_krnl_ldl(ctx)
    L_g = cl.Buffer(ctx, mf.WRITE_ONLY, L.nbytes)
    D_g = cl.Buffer(ctx, mf.WRITE_ONLY, D.nbytes)

    prg.ldl(queue, (1,), None, np.int32(A.shape[0]), np.int32(A.shape[1]), A_g, L_g, D_g)

    cl.enqueue_copy(queue, L, L_g)
    cl.enqueue_copy(queue, D, D_g)

    np.testing.assert_allclose(py_ldl_D, D, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(py_ldl_L[np.tril_indices(m)], L, rtol=1e-6, atol=1e-7)
