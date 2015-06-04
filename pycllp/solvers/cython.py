
from . import BaseSolver
import numpy as np

class CyHSDSolver(BaseSolver):
    name = 'cyhsd'

    def init(self, A, b, c, f=0.0):
        self.A = A.tocsc()
        if not self.A.has_sorted_indices:
            self.A.sort_indices()

        self.b = b
        self.c = c
        self.f = f

    def solve(self, ):
        from .._ipo import hsd_solver

        m,n = self.A.shape

        x = np.empty(n)
        y = np.empty(m)
        w = np.empty(m)
        z = np.empty(n)

        status = hsd_solver(m, n, self.A.nnz,
                            self.A.indices, self.A.indptr, self.A.data,
                            self.b, self.c,
                            self.f, x, y, w, z)
        self.x = x
        self.status = status
        return status
