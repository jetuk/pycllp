import numpy as np
from . import BaseSolver
from .cython_glpk import glpk_solve


class GlpkSolver(BaseSolver):
    name = 'glpk'

    def init(self, lp, verbose=0):
        pass

    def solve(self, lp, verbose=0):

        m, n, nlp = lp.nrows, lp.ncols, lp.nproblems
        c = np.array(lp.c, dtype=np.float64)
        b = np.array(lp.b, dtype=np.float64)
        # Offset for 1 based indexing
        ia = np.array(lp.A._rows, dtype=np.int32) + 1
        ja = np.array(lp.A._cols, dtype=np.int32) + 1
        ar = np.array(lp.A.data[0, :], dtype=np.float64)
        x = np.empty((nlp, n), dtype=np.float64)

        status = np.empty(nlp, dtype=np.int32)

        for i in range(nlp):
            status[i] = glpk_solve(ia, ja, ar, b[i, :], c[i, :], x[i, :])

        self.x = x
        self.status = status
        return status