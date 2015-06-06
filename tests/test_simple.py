"""
Simple test for different solvers of a small problem


"""
from __future__ import print_function
import numpy as np
import pytest
import pyopencl as cl
from pycllp.solvers import solver_registry
from itertools import product

non_cl_solvers = [(n,s) for n,s in solver_registry.items() if not n.startswith('cl')]
cl_solvers = [(n,s) for n,s in solver_registry.items() if n.startswith('cl')]
devices = [d for p in cl.get_platforms() for d in p.get_devices()]

def small_problem(fdtype=np.float64, idtype=np.int32):
    from scipy.sparse import csc_matrix
    A = np.array([3,2,2,5,1,3], dtype=fdtype)   # Only for "<" constraint equations!
    iA = np.array([0, 1, 0, 1, 0, 1], dtype=idtype)
    kA = np.array([0, 2, 4, 6], dtype=idtype)
    A = csc_matrix((A,iA,kA))

    b = np.array([10, 15], dtype=fdtype)       # Right hand side vector.
    c = np.array([2,3,4], dtype=fdtype)     # coefficients of variables in objective function.

    print(A.todense())
    return A, b, c


@pytest.mark.parametrize("name,solver_cls",non_cl_solvers)
def test_noncl_solvers(name, solver_cls):
    pytest_solver(name, solver_cls, [])

@pytest.mark.parametrize("device,name,solver_cls",
    [(d,n,s) for d,(n,s) in product(devices, cl_solvers)])
def test_cl_solvers(device, name, solver_cls):
    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)
    pytest_solver(name, solver_cls, [ctx, queue])

def pytest_solver(name, solver_cls, solver_args):
    from pycllp.lp import StandardLP
    lp = StandardLP(*small_problem())

    solver = solver_cls(*solver_args)
    lp.init(solver)
    status = lp.solve(solver)


    np.testing.assert_equal(solver.status, 0)
    np.testing.assert_almost_equal(np.squeeze(solver.x), (0.0,0.0,5.0))


"""

@pytest.fixture(params=)
def device(request):
    return request.param


def test_hsd_python():
    from pycllp.hsd import HSDLP
    args = small_problem()
    lp = HSDLP(*args[:-1] )
    status, x, y = lp.solve(args[-1])
    print('x',x)
    assert status == 0
    np.testing.assert_almost_equal(x, (0.0,0.0,5.0))

def test_hsd_cython():

    from pycllp._ipo import hsd_solver as solver
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

    from pycllp.cl.hsd import HSDLP_CL

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
"""
