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
        self.A = A

        self.b = b
        if self.b.ndim == 1:
            self.b = np.reshape(b, (1, len(b)))

        self.c = c
        if self.c.ndim == 1:
            self.c = np.reshape(c, (1, len(c)))

        # Number of simultaneous problems each to be solved in a
        # cl workgroup
        self.nlp = self.A.nproblems
        if np.isscalar(f):
            self.f = np.array([f, ])
        else:
            self.f = np.array(f)
        self.m, self.n = A.nrows, A.ncols


    def solve(self, ):
        raise NotImplementedError()


class BaseCSCSolver(BaseSolver):

    def init(self, A, b, c, f=0.0):
        BaseSolver.init(self, A, b, c, f=f)
        self.A, self.Ai, self.Ak = self.A.tocsc_arrays()


class BaseGeneralSolver(BaseSolver):
    def init(self, A, b, c, d, u, l, f=0.0):
        super(BaseGeneralSolver, self).init(A, b, c, f=f)
        self.d = d
        if self.d.ndim == 1:
            self.d = np.reshape(d, (1, len(d)))

        self.l = l
        if self.l.ndim == 1:
            self.l = np.reshape(l, (1, len(d)))

        self.u = u
        if self.u.ndim == 1:
            self.u = np.reshape(u, (1, len(d)))

# register solvers
from .cython import CyHSDSolver
from .hsd import HSDSolver
from .cl import ClHSDSolver
from .pathfollowing import DensePathFollowingSolver
