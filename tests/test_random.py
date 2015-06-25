import pytest
try:
    import pyopencl as cl
except ImportError:
    cl = None

from helpers import (non_cl_solvers, cl_solvers, devices, pytest_solver_parallel,
                     random_problem)
from itertools import product
from functools import partial

# Wrap random_problem to create a few tests cases of difference sizes
# Arguments for random_problem are m, n, density and nproblems
all_problems = (
    partial(random_problem, 10, 10, 0.1, 64),
    partial(random_problem, 100, 100, 0.1, 64),
    partial(random_problem, 250, 200, 0.1, 64),
)

@pytest.mark.noncl
@pytest.mark.parametrize("name,solver_cls,problem_func",
    [(n, s, pf) for (n, s), pf in product(non_cl_solvers,all_problems)])
def test_noncl_solvers(name, solver_cls, problem_func):
    pytest_solver_parallel(name, solver_cls, [], problem_func)

@pytest.mark.cl
@pytest.mark.parametrize("device,name,solver_cls,problem_func",
                         [(d, n, s, pf) for d, (n, s), pf in
                          product(devices, cl_solvers, all_problems)])
def test_cl_solvers_parallel(device, name, solver_cls, problem_func):
    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)
    pytest_solver_parallel(name, solver_cls, [ctx, queue], problem_func)
