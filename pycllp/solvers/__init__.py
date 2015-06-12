from six import with_metaclass
import numpy as np

solver_registry = {}

class MetaSolver(type):
    def __new__(cls, clsname, bases, attrs):
        newclass = super(MetaSolver, cls).__new__(cls, clsname, bases, attrs)
        if newclass.name is not None:
            solver_registry[newclass.name] = newclass
        return newclass

class BaseSolver(with_metaclass(MetaSolver)):
    name = None

    def init(self, A, b, c, f=0.0):
        self.A = A.tocsc()
        if not self.A.has_sorted_indices:
            self.A.sort_indices()

        self.b = b
        if self.b.ndim == 1:
            self.b = np.reshape(b, (1, len(b)))

        self.c = c
        if self.c.ndim == 1:
            self.c = np.reshape(c, (1, len(c)))

        # Number of simultaneous problems each to be solved in a
        # cl workgroup
        self.nlp = self.b.shape[0]
        self.f = f
        self.m,self.n = A.shape

    def solve(self, ):
        raise NotImplementedError()


# register solvers
from .cython import CyHSDSolver
from .hsd import HSDSolver
from .cl import ClHSDSolver
from .pathfollowing import DensePathFollowingSolver
