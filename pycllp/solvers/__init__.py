solver_registry = {}

class BaseSolver(object):
    class __metaclass__(type):
        def __init__(cls, name, bases, dct):
            type.__init__(cls, name, bases, dct)
            if cls.name is not None:
                solver_registry[cls.name] = cls
    name = None

    def init(self, A, b, c, f=0.0):
        raise NotImplementedError()

    def solve(self, ):
        raise NotImplementedError()


# register solvers
from cython import CyHSDSolver
from hsd import HSDSolver
from cl import ClHSDSolver
