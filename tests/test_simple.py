"""
Simple test for different solvers of a small problem


"""
from __future__ import print_function
import numpy as np
import pytest
import pyopencl as cl


def small_problem(fdtype=np.float64, idtype=np.int32):
    c = np.array([2,3,4], dtype=fdtype)     # coefficients of variables in objective function.
    A = np.array(
                [[3,2,1],   # Coefficient matrix,
                [2,5,3]], dtype=fdtype)   # Only for "<" constraint equations!

    b = np.array([10, 15], dtype=fdtype)       # Right hand side vector.
    m, n = A.shape
    A = A.T.flatten()
    nz = len(A)
    iA = np.array([0, 1, 0, 1, 0, 1], dtype=idtype)
    kA = np.array([0, 2, 4, 6], dtype=idtype)
    f = 0.001

    return m, n, nz, iA, kA, A, b, c, f


def test_hsd_python():
    from pyipo.hsd import HSDLP
    args = small_problem()
    lp = HSDLP(*args[:-1] )
    status, x, y = lp.solve(args[-1])
    print('x',x)
    assert status == 0
    np.testing.assert_almost_equal(x, (0.0,0.0,5.0))

def test_hsd_cython():

    from pyipo._ipo import hsd_solver as solver
    args = small_problem()
    n = args[1]
    x = np.empty(n)
    y = np.empty(n)
    w = np.empty(n)
    z = np.empty(n)
    args = args+(x, y, w, z)
    status = solver(*args)
    print('x',x)
    assert status == 0
    np.testing.assert_almost_equal(x, (0.0,0.0,5.0))

@pytest.fixture(params=[d for p in cl.get_platforms() for d in p.get_devices()])
def device(request):
    return request.param

def test_hsd_cl(device, ):

    from pyipo.cl.hsd import HSDLP_CL

    m, n, nz, iA, kA, A, b, c, f = small_problem(fdtype=np.float32,
                                                 idtype=np.int32)

    b = b.reshape((1,len(b)))
    c = c.reshape((1,len(c)))

    print('Creating CL context and queue on device: {}...'.format(device.name))
    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)

    lp = HSDLP_CL(m, n, nz, iA, kA, A, b, c )
    lp.init_cl(ctx,)
    status, x, y = lp.solve(ctx, queue, f)
    assert status[0] == 0
    np.testing.assert_almost_equal(x[0,:], (0.0,0.0,5.0), decimal=5)
