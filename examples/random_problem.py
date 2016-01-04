#!/usr/bin/python
"""
An example that runs a random problem.

"""
import numpy as np
import pyopencl as cl
from pycllp.solvers import solver_registry
from pycllp.lp import StandardLP


def random_problem(m, n, density, nproblems, seed=0):
    """
    Generate a random problem with m rows and n columns and nproblems.
    Sparse matrix with density generated using scipy.sparse.rand
    """
    from scipy.sparse import rand
    from pycllp.lp import SparseMatrix

    np.random.seed(seed)
    A = SparseMatrix(matrix=rand(m, n, density=density))
    m = A.nrows
    n = A.ncols
    b = (0.5+np.random.rand(nproblems, m))
    c = (0.5+np.random.rand(nproblems, n))

    return A, b, c, 0.0


def solve(N, NP, solver_name):
    try:
        solver_cls = solver_registry[solver_name]
    except KeyError:
        raise ValueError('Solver {} not recognised.'.format(solver_name))

    solver_args = []
    if solver_name.startswith('cl'):
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        solver_args.extend([ctx, queue])

    args = random_problem(N, N, 0.1, NP)

    slp = StandardLP(*args)
    slp = slp.to_equality_form()

    solver = solver_cls(*solver_args)
    slp.init(solver)
    slp.solve(solver, verbose=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Execute a randomly generated problem.')
    parser.add_argument('n', type=int, help='Number of columns in random problem.')
    parser.add_argument('nlp', type=int, help='Number of parallel problems.')
    parser.add_argument('solver', type=str, help='Name of solver to use.')

    args = parser.parse_args()
    solve(args.n, args.nlp, args.solver)
