import pytest
import numpy as np
import vanderbei_problems
from helpers import non_cl_solvers, cl_solvers, devices
import inspect
from itertools import product

all_problems = inspect.getmembers(vanderbei_problems, inspect.isfunction)
print(all_problems)

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
