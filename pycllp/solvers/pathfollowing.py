"""
Original implementation:
    Implementation of the Primal-Dual Interior Point Method
        Robert J. Vanderbei
        28 November 1994
        http://www.princeton.edu/~rvdb/LPbook/

Port to Python
    James E. Tomlinson
    2015
"""
from __future__ import print_function
import numpy as np
from . import BaseSolver
import sys

EPS = 1.0e-6
MAX_ITER = 200


class DensePathFollowingSolver(BaseSolver):
    """
    This solver is a port of the path following algorith originally
    implemented by Vanderbei. It is for solving LP in the standard
    form.
    """
    name = 'dense_path_following'

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
        m2 = m+n
        A = np.array(lp.A.todense(problem=ilp))
        b = lp.b[ilp, :]
        c = lp.c[ilp, :]
        f = lp.f[ilp]

        normr, norms = 0.0, 0.0  # infeasibilites
        gamma, delta, mu, theta = 0.0, 0.0, 0.0, 0.0  # parameters
        status = 5
        primal_obj, dual_obj = 0.0, 0.0

        # Allocate memory for arrays.

        # Step direction
        dx = np.zeros(n)
        dxy = np.zeros(m2)
        dy = np.zeros(m)
        dz = np.zeros(n)
        # infeasibilites
        rho = np.zeros(m)
        sigma = np.zeros(n)
        # Diagonal matrixes
        B = np.zeros((m2, m2))
        D = np.zeros(n)
        E = np.zeros(m)

        #  Initialization.

        x = np.ones(n)
        z = np.ones(n)
        y = np.ones(m)

        normr0 = sys.float_info.max
        norms0 = sys.float_info.max

        delta = 0.02
        r = 0.9

        At = A.T

        B[:m, m:] = A   # top right
        B[m:, :m] = At  # bottom left

        # 	Display Banner.
        if verbose > 0:
            print(A)
            print(b)
            print(c)
            print("m = {:d},n = {:d},nz = {:d}".format(m, n, lp.A.nnzeros))
            print(
            """---------------------------------------------------------------
                     |           Primal          |            Dual           |
              Iter   |  Obj Value       Infeas   |  Obj Value       Infeas   |
            - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            """)

        # 	Iteration.

        for _iter in range(MAX_ITER):
            # STEP 1: Compute infeasibilities.

            rho[...] = b - np.dot(A, x)
            normr = np.sqrt(np.dot(rho, rho))

            sigma[...] = c - np.dot(At, y) + z
            norms = np.sqrt(np.dot(sigma, sigma))

            # STEP 2: Compute duality gap.
            gamma = np.dot(z, x)

            # Print statistics.
            primal_obj = np.dot(c, x) + f
            dual_obj = np.dot(b, y) + f
            if verbose > 0:
                print("{:8d}   {:14.7e}  {:8.1e}    {:14.7e}  {:8.1e} {:8.1e} {:8.1e} ".format(
                  _iter, primal_obj, normr, dual_obj, norms, gamma, EPS))

            # STEP 2.5: Check stopping rule.

            if normr < EPS and norms < EPS and gamma < EPS:
                status = 0
                break  # OPTIMAL

            if normr > 10*normr0:
                status = 2
                break  # PRIMAL INFEASIBLE (unreliable)

            if norms > 10*norms0:
                status = 4
                break  # DUAL INFEASIBLE (unreliable)

            # STEP 3: Compute central path parameter.

            mu = delta * gamma / n

            # STEP 4: Compute step directions.
            D[...] = z/x
            E[...] = 0.0

            # Create B matrix
            B[:m, :m] = np.diag(E)  # top left
            B[m:, m:] = np.diag(D)  # bottom right

            dx = sigma - z + mu/x
            dy = rho
            dxy[:m] = dy
            dxy[m:] = dx
            dxy = np.linalg.solve(B, dxy)
            dy = dxy[:m]
            dx = dxy[m:]

            dz = mu/x - z - D*dx

            # STEP 5: Ratio test to find step length.

            theta = max(0.0, np.max(-dx/x), np.max(-dz/z), np.max(-dy/y))
            theta = min(r/theta, 1.0)

            # STEP 6: Step to new point

            x += theta*dx
            z += theta*dz
            y += theta*dy

        normr0 = normr
        norms0 = norms

        self.status[ilp] = status
        self.x[ilp, :] = x
