
from . import BaseCSCSolver
import numpy as np

class CyHSDSolver(BaseCSCSolver):
    name = 'cyhsd'

    def init(self, Ai, Aj, Adata, b, c, f=0.0):
        BaseCSCSolver.init(self, Ai, Aj, Adata, b, c, f=f)

    def solve(self, verbose=0):
        from .._ipo import hsd_solver

        m,n,nlp = self.m,self.n,self.nlp

        x = np.empty((nlp,n))
        y = np.empty((nlp,m))
        w = np.empty((nlp,m))
        z = np.empty((nlp,n))
        status = np.empty(nlp, dtype=np.int)

        for i in range(nlp):
            status[i] = hsd_solver(m, n, self.A.nnz,
                            self.A.indices, self.A.indptr, self.Adata[i,:],
                            self.b[i,:], self.c[i,:],
                            self.f[i], x[i,:], y[i,:], w[i,:], z[i,:],
                            verbose)


        self.x = x
        self.status = status
        return status
