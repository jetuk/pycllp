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
        self.nlp = self.b.shape[0]
        self.f = np.array(f)
        self.m, self.n = A.nrows, A.ncols

    def solve(self, ):
        raise NotImplementedError()


class BaseCSCSolver(BaseSolver):

    def init(self, A, b, c, f=0.0):
        BaseSolver.init(self, A, b, c, f=f)
        self.Adata = self.A.data
        self.A = self.A.tocsc()

# register solvers
try:
    from .cython import CyHSDSolver
except ImportError:
    pass

try:
    from .hsd import HSDSolver
except ImportError:
    pass

try:
    from .cl import ClHSDSolver
except ImportError:
    pass

try:
    from .pathfollowing import DensePathFollowingSolver
except ImportError:
    pass
