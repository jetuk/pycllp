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
    return 100


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
