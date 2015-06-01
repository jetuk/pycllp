"""
Simple test for different solvers of a small problem


"""
from __future__ import print_function
import numpy as np
import pytest
import pyopencl as cl


def small_problem(fdtype=np.float, idtype=np.int):
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

    # Perturb the c and b matrices
    N = 128
    B = np.ones((N,len(b))).astype(fdtype)*b
    C = (0.5+np.random.rand(N,len(c)).astype(fdtype))*c
    print(B.shape, C.shape)
    return m, n, nz, iA, kA, A, B, C, f

def solve_hsd_python(*args):
    from pycllp.hsd import HSDLP
    lp = HSDLP(*args[:-1] )
    status, x, y = lp.solve(args[-1])

    assert status == 0
    return status, x

@pytest.fixture(params=[d for p in cl.get_platforms() for d in p.get_devices()])
def device(request):
    return request.param

def test_hsd_cl(device, ):
    from pycllp.cl.hsd import HSDLP_CL

    m, n, nz, iA, kA, A, b, c, f = small_problem(fdtype=np.float32,
                                                 idtype=np.int32)

    print('Creating CL context and queue on device: {}...'.format(device.name))
    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)

    lp = HSDLP_CL(m, n, nz, iA, kA, A, b, c )
    lp.init_cl(ctx,)
    status, x, y = lp.solve(ctx, queue, f)

    for i in range(len(status)):
        print("Problem {:d} finished with status: {:d}".format(i, status[i]))
        # solve with cython version
        cystat, cyx = solve_hsd_python(m, n, nz, iA, kA, A, b[i,:], c[i,:], f)
        np.testing.assert_almost_equal(x[i,:], cyx, decimal=4)
        assert cystat == status[i]
