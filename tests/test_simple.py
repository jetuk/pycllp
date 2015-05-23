"""
Simple test for different solvers of a small problem


"""
from __future__ import print_function

import pytest

@pytest.fixture
def small_problem():
    import numpy as np
    c = np.array([2,3,4], dtype=np.float64)     # coefficients of variables in objective function.
    A = np.array(
                [[3,2,1],   # Coefficient matrix,
                [2,5,3]], dtype=np.float64)   # Only for "<" constraint equations!

    b = np.array([10, 15], dtype=np.float64)       # Right hand side vector.
    m, n = A.shape
    A = A.T.flatten()
    nz = len(A)
    iA = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    kA = np.array([0, 2, 4, 6], dtype=np.int32)
    f = 0.001

    return m, n, nz, iA, kA, A, b, c, f


def test_hsd_python():
    from pyipo.hsd import solver
    args = small_problem()

    status, x, y = solver(*args)
    assert status == 0

def test_hsd_cython():
    import numpy as np
    from pyipo._ipo import hsd_solver as solver
    args = small_problem()
    n = args[1]
    x = np.empty(n)
    y = np.empty(n)
    w = np.empty(n)
    z = np.empty(n)
    args = args+(x, y, w, z)
    status = solver(*args)
    assert status == 0
