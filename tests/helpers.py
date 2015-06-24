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
    old_A_data = A.data.copy()
    A.set_num_problems(N)
    A.data = np.ones(A.data.shape)*old_A_data

    return A, b, c, r, l, u, f
