from six import with_metaclass

solver_registry = {}


class MetaSolver(type):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(MetaSolver, cls).__new__(cls, clsname, bases, attrs)
        if newclass.name is not None:
            solver_registry[newclass.name] = newclass
        return newclass


class BaseSolver(with_metaclass(MetaSolver)):
    name = None

    def init(self, lp, verbose=0):
        raise NotImplementedError()

    def solve(self, lp, verbose=0):
        raise NotImplementedError()


class BaseCSCSolver(BaseSolver):
    def init(self, lp, verbose=0):
        self.A, self.Ai, self.Ak = lp.A.tocsc_arrays()


class BaseGeneralSolver(BaseSolver):
    pass


# register solvers
# Cython HSD solver is for standard LPs, not equality LPs for which this library & tests
# are current setup. TODO fix this.
#from .cython import CyHSDSolver
#from .hsd import HSDSolver
from .cl import ClDensePrimalNormalSolver, ClSparsePrimalNormalSolver
from .pathfollowing import DensePathFollowingSolver
from .normal_eqns import DensePrimalNormalSolver
from .glpk import GlpkSolver