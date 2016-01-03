"""
This module contains interior point based solvers that use the so called
normal equations to solve the central path.
"""
from __future__ import print_function
import numpy as np
from scipy.linalg import solve_triangular
from . import BaseSolver
from .._ldl import solve_primal_normal
import sys

EPS = 1.0e-8
MAX_ITER = 200


class DensePrimalNormalSolver(BaseSolver):
    """

    """
    name = 'dense_primal_normal'

    def init(self, lp, verbose=0):
        pass

    def solve(self, lp, verbose=0):
        n, nlp = lp.ncols, lp.nproblems
        self.x = np.empty((nlp, n))
        self.status = np.empty(nlp, dtype=np.int)

        for i in range(nlp):
            self._solve(lp, i, verbose=verbose)

        return self.status

    def _solve(self, lp, ilp, verbose=0):
        n = lp.ncols
        m = lp.nrows

        A = np.array(lp.A.todense(problem=ilp))
        b = lp.b[ilp, :]
        c = lp.c[ilp, :]
        f = lp.f[ilp]

        x = np.ones(n)
        z = np.ones(n)
        y = np.ones(m)


        normr0 = sys.float_info.max
        norms0 = sys.float_info.max

        delta = 0.1
        r = 0.9

        status = 5

        for _iter in range(MAX_ITER):
            rho = b - np.dot(A, x)
            normr = np.sqrt(np.dot(rho, rho))

            sigma = c - np.dot(A.T, y) + z
            norms = np.sqrt(np.dot(sigma, sigma))

            gamma = np.dot(z, x)
            mu = delta * gamma / n

            if verbose > 0:
                print("{:d} |rho|: {:8.1e}  |sigma| {:8.1e}  gamma: {:8.1e}".format(_iter, normr, norms, gamma))

            if normr < EPS and norms < EPS and gamma < EPS:
                status = 0
                break  # OPTIMAL

            if normr > 10*normr0 and normr > EPS:
                status = 2
                break  # PRIMAL INFEASIBLE (unreliable)

            if norms > 10*norms0 and norms > EPS:
                status = 4
                break  # DUAL INFEASIBLE (unreliable)

            # Create system of primal normal equations
            dy = solve_primal_normal(A, x, z, y, b, c, mu, delta=1e-6)

            if np.any(np.isnan(dy)):
                status = 3
                break

            dx = (c - A.T.dot(y) + mu/x - A.T.dot(dy))*x/z
            dz = (mu - x*z - z*dx)/x

            theta = max(np.max(-dx/x), np.max(-dz/z))
            theta = min(r/theta, 1.0)

            x += theta*dx
            z += theta*dz
            y += theta*dy

            normr0 = normr
            norms0 = norms

        self.status[ilp] = status
        self.x[ilp, :] = x
