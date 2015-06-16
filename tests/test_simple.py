"""
Simple test for different solvers of a small problem


"""
from __future__ import print_function
import numpy as np
import pytest
import pyopencl as cl
from helpers import non_cl_solvers, cl_solvers, devices
from itertools import product


def small_problem():
    from scipy.sparse import csc_matrix
    A = np.array([3,2,2,5,1,3], dtype=np.float)   # Only for "<" constraint equations!
    iA = np.array([0, 1, 0, 1, 0, 1], )
    kA = np.array([0, 2, 4, 6],)
    A = csc_matrix((A,iA,kA)).tocoo()

    b = np.array([10, 15], dtype=np.float)       # Right hand side vector.
    c = np.array([2,3,4], dtype=np.float )     # coefficients of variables in objective function.

    c = np.array([ 1.10685436,  3.67678309,  2.04570983])
    b = np.array([  5.187898,    16.76453246])


    return A.row, A.col, A.data, b, c


def parallel_small_problem(N=1024):
    """
    Take small_problem and perturb randomly to generate N problems
    """
    Ai, Aj, Adata, b, c = small_problem()
    np.random.seed(0)
    b = (0.5+np.random.rand( N,len(b) ))*b
    c = (0.5+np.random.rand( N,len(c) ))*c

    return Ai, Aj, Adata, b, c

@pytest.mark.noncl
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
    status = lp.solve(solver, verbose=2)


    np.testing.assert_equal(solver.status, 0)
    np.testing.assert_almost_equal(np.squeeze(solver.x), (  1.00997e-13,   1.22527e-12,   5.18790e+00),
        decimal=5)


@pytest.mark.parametrize("device,name,solver_cls",
    [(d,n,s) for d,(n,s) in product(devices, cl_solvers)])
def test_cl_solvers_parallel(device, name, solver_cls):
    print("Device", device)
    from pycllp.lp import StandardLP
    from pycllp.solvers import solver_registry

    args = parallel_small_problem()
    lp = StandardLP(*args)
    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)

    solver = solver_cls(ctx, queue)
    lp.init(solver)
    lp.solve(solver)

    pysolver = solver_registry['cyhsd']()
    lp.init(pysolver)
    lp.solve(pysolver)

    np.testing.assert_almost_equal(solver.status, pysolver.status,)
    np.testing.assert_almost_equal(
                np.sum(solver.x*c,axis=1),
                np.sum(pysolver.x*c,axis=1).astype(np.float32), decimal=4)
