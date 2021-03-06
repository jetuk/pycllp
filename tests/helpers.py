import numpy as np
try:
    import pyopencl as cl
except ImportError:
    cl = None
from pycllp.solvers import solver_registry

non_cl_solvers = [(n, s) for n, s in solver_registry.items()
                  if not n.startswith('cl')]
devices = []
cl_solvers = []
if cl:
    cl_solvers = [(n, s) for n, s in solver_registry.items() if n.startswith('cl')]
    devices = [d for p in cl.get_platforms() for d in p.get_devices()]


def perturb_problem(problem_func, N):
    """
    Perturb a standard problem function to create random variations.
    """
    A, b, c, r, l, u, f, xopt = problem_func()

    np.random.seed(0)
    b = (0.5+np.random.rand(N, len(b)))*b
    r = (0.5+np.random.rand(N, len(r)))*r
    c = (0.5+np.random.rand(N, len(c)))*c

    #old_A_data = A.data.copy()
    #A.set_num_problems(N)
    #A.data = np.ones(A.data.shape)*old_A_data

    return A, b, c, r, l, u, f


def random_problem(m, n, density, nproblems):
    """
    Generate a random problem with m rows and n columns and nproblems.
    Sparse matrix with density generated using scipy.sparse.rand
    """
    from scipy.sparse import rand, csc_matrix
    from pycllp.lp import SparseMatrix, StandardLP

    np.random.seed(0)

    A = np.empty((m, n))
    for i in range(m):
        A[i, :] = rand(1, n, density=max(density, 3./n)).todense()

    A = SparseMatrix(matrix=csc_matrix(A))
    m = A.nrows
    n = A.ncols
    b = np.random.rand(nproblems, m)
    c = np.random.rand(nproblems, n)

    # Create sparse matrix with scipy.sparse
    #A.set_num_problems(nproblems)
    # TODO make this random.
    #A.data = np.ones(A.data.shape)*old_A_data

    return StandardLP(A, b, c, 0.0)


def pytest_solver_parallel(name, solver_cls, solver_args, problem_func,
                           compare_with_solver='glpk'):
    """
    Test problem_func with solver_cls using solver_args against another solver.
    """
    from pycllp.lp import GeneralLP, StandardLP
    from pycllp.solvers import solver_registry

    lp = problem_func()
    try:
        if len(lp) == 2:
            lp, xopt = lp
            if xopt.ndim == 1:
                xopt = np.expand_dims(xopt, axis=0)
    except TypeError:
        xopt = None

    if isinstance(lp, StandardLP):
        elp = lp.to_equality_form()
    else:
        elp = lp

    solver = solver_cls(*solver_args)
    elp.init(solver)
    elp.solve(solver, verbose=2)
    npblms = elp.nproblems

    if xopt is None:
        solver2 = solver_registry[compare_with_solver]()
        elp.init(solver2)
        elp.solve(solver2, verbose=0)
        # Test that the optimal solutions are the same
        xopt = solver2.x

    assert np.all(solver.status==0)
    for i in range(npblms):
        if solver.status[i] == 0:
            # Test only the non-slack variables
            ind = np.where(np.abs(elp.c[i, :]) > 0.0)
            np.testing.assert_allclose(xopt[i, ind], solver.x[i, ind], rtol=1e-3, atol=1e-3)
