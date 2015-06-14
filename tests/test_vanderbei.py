import pytest
import numpy as np
import pyopencl as cl
import vanderbei_problems
from helpers import non_cl_solvers, cl_solvers, devices, perturb_problem
import inspect
from itertools import product

all_problems = inspect.getmembers(vanderbei_problems, inspect.isfunction)

@pytest.mark.noncl
@pytest.mark.parametrize("name,solver_cls,problem_name,problem_func",
    [(n, s, pn, pf) for (n, s), (pn, pf) in product(non_cl_solvers,all_problems)])
def test_noncl_solvers(name, solver_cls, problem_name, problem_func):
    pytest_solver(name, solver_cls, [], problem_func)


def pytest_solver(name, solver_cls, solver_args, problem_func):
    print("Testing Vanderbei {} with solver {}...".format(problem_func, name))
    from pycllp.lp import GeneralLP

    args = list(problem_func())
    xopt = args.pop()
    lp = GeneralLP(*args)
    slp = lp.to_standard_form()

    solver = solver_cls(*solver_args)
    slp.init(solver)
    slp.solve(solver, verbose=2)

    np.testing.assert_equal(solver.status, 0)
    np.testing.assert_almost_equal(np.squeeze(solver.x), xopt,
                                   decimal=5)


@pytest.mark.parametrize("device,name,solver_cls,problem_func",
                         [(d, n, s, pf) for d, (n, s), (pn, pf) in
                          product(devices, cl_solvers, all_problems)])
def test_cl_solvers_parallel(device, name, solver_cls, problem_func):
    from pycllp.lp import GeneralLP
    from pycllp.solvers import solver_registry

    args = perturb_problem(problem_func, 1024)

    lp = GeneralLP(*args)
    slp = lp.to_standard_form()

    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)

    solver = solver_cls(ctx, queue)
    slp.init(solver)
    slp.solve(solver, verbose=2)

    pysolver = solver_registry['cyhsd']()
    slp.init(pysolver)
    slp.solve(pysolver, verbose=2)

    np.testing.assert_almost_equal(solver.status, pysolver.status,)
    np.testing.assert_almost_equal(
                solver.x,
                pysolver.x, decimal=4)
