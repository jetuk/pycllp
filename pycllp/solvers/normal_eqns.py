"""
This module contains interior point based solvers that use the so called
normal equations to solve the central path.
"""
from __future__ import print_function
import numpy as np
from scipy.linalg import solve_triangular
from . import BaseSolver
from ..ldl import modified_ldl, solve_primal_normal, refined_forward_backward, modified_cholesky, forward_backward


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
        print(ilp)
        n = lp.ncols
        m = lp.nrows
        m2 = m+n
        A = np.array(lp.A.todense(problem=ilp))
        b = lp.b[ilp, :]
        c = lp.c[ilp, :]
        f = lp.f[ilp]

        x = np.ones(n)
        z = np.ones(n)
        w = np.ones(m)
        y = np.ones(m)

        import sys
        normr0 = sys.float_info.max
        norms0 = sys.float_info.max

        delta = 0.02
        r = 0.9

        status = 5

        for _iter in range(MAX_ITER):
            rho = b - np.dot(A, x) - w
            #normr = np.sum(np.abs(rho))
            normr = np.sqrt(np.dot(rho, rho))
            sigma = c - np.dot(A.T, y) + z
            #norms = np.sum(np.abs(sigma))
            norms = np.sqrt(np.dot(sigma, sigma))
            gamma = np.dot(z, x) + np.dot(w, y)
            mu = delta * gamma / (n+m)

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
            dy2 = solve_primal_normal(A, x, z, y, w, b, c, mu, delta=1e-8)

            AA = np.eye(m)*w/y + (A*x/z).dot(A.T)

            bb = b - A.dot(x) - mu/y - (A*x/z).dot(c - A.T.dot(y) + mu/x)

            L, skipped = modified_cholesky(AA)
            dy = forward_backward(L, L.T, -bb)
            print(skipped)

            # Solve the system for dy
            #dy = forward_backward_modified_ldl(AA, -bb)

            print(np.abs(dy2-dy))

            if np.any(np.isnan(dy)):
                status = 2
                break
            #try:
            #    dy = np.squeeze(np.linalg.solve(-AA, bb))
            #except np.linalg.LinAlgError:
            #    status = 2
            #    break
            dx = (c - A.T.dot(y) + mu/x - A.T.dot(dy))*x/z
            dz = (mu - x*z - z*dx)/x
            dw = (mu - y*w - w*dy)/y

            theta = max(np.max(-dx/x), np.max(-dz/z),
                        np.max(-dy/y), np.max(-dw/w))
            theta = min(r/theta, 1.0)

            x += theta*dx
            z += theta*dz
            y += theta*dy
            w += theta*dw

            normr0 = normr
            norms0 = norms

        self.status[ilp] = status
        self.x[ilp, :] = x
