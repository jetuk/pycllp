from six import with_metaclass

solver_registry = {}

class MetaSolver(type):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(MetaSolver, cls).__new__(cls, clsname, bases, attrs)
        solver_registry[newclass.name] = newclass
        return newclass

class BaseSolver(with_metaclass(MetaSolver)):
    name = None
    
    def init(self, A, b, c, f=0.0):
        raise NotImplementedError()

    def solve(self, ):
        raise NotImplementedError()


# register solvers
from .cython import CyHSDSolver
from .hsd import HSDSolver
from .cl import ClHSDSolver
