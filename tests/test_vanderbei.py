import pytest
import numpy as np
try:
    import pyopencl as cl
except ImportError:
    cl = None
import vanderbei_problems
from helpers import (non_cl_solvers, cl_solvers, devices, pytest_solver_parallel,
                     perturb_problem)
import inspect
from itertools import product
from functools import partial

all_problems = inspect.getmembers(vanderbei_problems, inspect.isfunction)
# Create perturbed problems
#all_problems = [(pn, partial(perturb_problem, pf, 32)) for pn, pf in all_problems]

@pytest.mark.noncl
@pytest.mark.parametrize("name,solver_cls,problem_name,problem_func",
    [(n, s, pn, pf) for (n, s), (pn, pf) in product(non_cl_solvers,all_problems)])
def test_noncl_solvers(name, solver_cls, problem_name, problem_func):
    pytest_solver(name, solver_cls, [], problem_func)


def pytest_solver(name, solver_cls, solver_args, problem_func):
    print("Testing Vanderbei {} with solver {}...".format(problem_func, name))
    from pycllp.lp import StandardLP

    lp, xopt = problem_func()
    ncols = lp.ncols
    print(ncols)
    if isinstance(lp, StandardLP):
        elp = lp.to_equality_form()
    else:
        elp = lp

    solver = solver_cls(*solver_args)
    elp.init(solver)
    elp.solve(solver, verbose=2)

    np.testing.assert_equal(solver.status, 0)
    np.testing.assert_allclose(solver.x[0, :ncols], xopt, rtol=1e-6, atol=1e-6)

@pytest.mark.cl
@pytest.mark.parametrize("device,name,solver_cls,problem_func",
                         [(d, n, s, pf) for d, (n, s), (pn, pf) in
                          product(devices, cl_solvers, all_problems)])
def test_cl_solvers_parallel(device, name, solver_cls, problem_func):
    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)
    pytest_solver_parallel(name, solver_cls, [ctx, queue], problem_func)
