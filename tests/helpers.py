
import pyopencl as cl
from pycllp.solvers import solver_registry

non_cl_solvers = [(n, s) for n, s in solver_registry.items() if not n.startswith('cl')]
cl_solvers = [(n, s) for n, s in solver_registry.items() if n.startswith('cl')]
devices = [d for p in cl.get_platforms() for d in p.get_devices()]
