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
    print(A.nproblems)
    #old_A_data = A.data.copy()
    #A.set_num_problems(N)
    #A.data = np.ones(A.data.shape)*old_A_data

    return A, b, c, r, l, u, f


def random_problem(m, n, density, nproblems):
    """
    Generate a random problem with m rows and n columns and nproblems.
    Sparse matrix with density generated using scipy.sparse.rand
    """
    from scipy.sparse import rand
    from pycllp.lp import SparseMatrix

    np.random.seed(0)
    A = SparseMatrix(matrix=rand(m, n, density=density))
    m = A.nrows
    n = A.ncols
    b = (0.5+np.random.rand(nproblems, m))
    c = (0.5+np.random.rand(nproblems, n))

    # Create sparse matrix with scipy.sparse

    old_A_data = A.data.copy()
    #A.set_num_problems(nproblems)
    # TODO make this random.
    #A.data = np.ones(A.data.shape)*old_A_data

    return A, b, c, 0.0


def pytest_solver_parallel(name, solver_cls, solver_args, problem_func,
                           compare_with_solver='cyhsd'):
    """
    Test problem_func with solver_cls using solver_args against another solver.
    """
    from pycllp.lp import GeneralLP, StandardLP
    from pycllp.solvers import solver_registry

    args = problem_func()
    if len(args) == 4:
        slp = StandardLP(*args)
    else:
        lp = GeneralLP(*args)
        slp = lp.to_standard_form()

    solver = solver_cls(*solver_args)
    slp.init(solver)
    slp.solve(solver, verbose=0)

    pysolver = solver_registry[compare_with_solver]()
    slp.init(pysolver)
    slp.solve(pysolver, verbose=0)
    ind = np.where(solver.status == 0)
    np.testing.assert_almost_equal(solver.status==0, pysolver.status==0,)
    np.testing.assert_almost_equal(
                solver.x[ind, :],
                pysolver.x[ind, :], decimal=2)
