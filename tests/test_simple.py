"""
Simple test for different solvers of a small problem


"""
from __future__ import print_function
import numpy as np
import pytest
try:
    import pyopencl as cl
except ImportError:
    pass
from helpers import non_cl_solvers, cl_solvers, devices
from itertools import product
from pycllp.lp import SparseMatrix, StandardLP

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

    return SparseMatrix(matrix=A), b, c, 0.0


def parallel_small_problem(N=32):
    """
    Take small_problem and perturb randomly to generate N problems
    """
    A, b, c, f = small_problem()
    np.random.seed(0)
    b = (0.5+np.random.rand(N, len(b)))*b
    c = (0.5+np.random.rand(N, len(c)))*c

    return A, b, c, f


@pytest.mark.noncl
@pytest.mark.parametrize("name,solver_cls",non_cl_solvers)
def test_noncl_solvers(name, solver_cls):
    pytest_solver(name, solver_cls, [])

@pytest.mark.cl
@pytest.mark.parametrize("name,solver_cls", cl_solvers)
def test_cl_solvers(name, solver_cls):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    pytest_solver(name, solver_cls, [ctx, queue])

def pytest_solver(name, solver_cls, solver_args):
    lp = StandardLP(*small_problem())
    ncols = lp.ncols
    lp = lp.to_equality_form()

    solver = solver_cls(*solver_args)
    lp.init(solver, verbose=2)
    status = lp.solve(solver, verbose=2)

    np.testing.assert_equal(solver.status, 0)
    np.testing.assert_allclose(solver.x[0, :ncols], (1.00997e-13,   1.22527e-12,   5.18790e+00), rtol=1e-1, atol=1e-1)


@pytest.mark.cl
@pytest.mark.parametrize("name,solver_cls", cl_solvers)
def test_cl_solvers_parallel(name, solver_cls):
    from pycllp.solvers import solver_registry

    args = parallel_small_problem()
    lp = StandardLP(*args)
    lp = lp.to_equality_form()
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    cl_size = lp.nproblems

    solver = solver_cls(ctx, queue)
    lp.init(solver, verbose=0)
    lp.solve(solver, verbose=0)

    pysolver = solver_registry['dense_primal_normal']()
    lp.init(pysolver)
    lp.solve(pysolver, verbose=0)

    np.testing.assert_allclose(solver.status, pysolver.status,)
    for i in range(cl_size):
        np.testing.assert_allclose(pysolver.x[i, :], solver.x[i, ...],
                                   rtol=1e-3, atol=1e-3)
