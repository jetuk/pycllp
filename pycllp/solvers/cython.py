
from . import BaseCSCSolver
from .._ipo import hsd_solver
import numpy as np

class CyHSDSolver(BaseCSCSolver):
    name = 'cyhsd'

    def solve(self, lp, verbose=0):

        m, n, nlp = lp.nrows, lp.ncols, lp.nproblems
        b = np.nan_to_num(lp.b)

        x = np.empty((nlp,n))
        y = np.empty((nlp,m))
        w = np.empty((nlp,m))
        z = np.empty((nlp,n))
        status = np.empty(nlp, dtype=np.int32)

        for i in range(nlp):
            status[i] = hsd_solver(m, n, len(self.Ai),
                            self.Ai, self.Ak, self.A[0,:],
                            b[i,:], lp.c[i,:],
                            lp.f[i], x[i,:], y[i,:], w[i,:], z[i,:],
                            verbose)


        self.x = x
        self.status = status
        return status
