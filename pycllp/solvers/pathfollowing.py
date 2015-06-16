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


EPS = 1.0e-12
MAX_ITER = 200


class DensePathFollowingSolver(BaseSolver):
    name = 'dense_path_following'

    def init(self, A, b, c, f=0.0):
        BaseSolver.init(self, A, b, c, f=f)

    def solve(self, verbose=0):
        n, nlp = self.n, self.nlp
        self.x = np.empty((nlp, n))
        self.status = np.empty(nlp, dtype=np.int)

        for i in range(nlp):
            self._solve(i, verbose=verbose)

        return self.status

    def _solve(self, ilp, verbose=0):
        n = self.n
        m = self.m
        m2 = m+n
        A = self.A.todense(problem=ilp)
        b = self.b[ilp, :]
        c = self.c[ilp, :]
        f = self.f[ilp]

        normr, norms = 0.0, 0.0  # infeasibilites
        gamma, delta, mu, theta = 0.0, 0.0, 0.0, 0.0  # parameters
        status = 5
        primal_obj, dual_obj = 0.0, 0.0

        # Allocate memory for arrays.

        # Step direction
        dx = np.zeros(n)
        dw = np.zeros(m)
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
        w = np.ones(m)
        y = np.ones(m)
        import sys
        normr0 = sys.float_info.max
        norms0 = sys.float_info.max

        delta = 0.02
        r = 0.9

        At = A.T

        # 	Display Banner.
        if verbose > 0:
            print("m = {:d},n = {:d},nz = {:d}".format(m, n, self.A.nnzeros))
            print(
            """--------------------------------------------------------------------------
                     |           Primal          |            Dual           |       |
              Iter   |  Obj Value       Infeas   |  Obj Value       Infeas   |  mu   |
            - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            """)

        # 	Iteration.

        for _iter in range(MAX_ITER):
            # STEP 1: Compute infeasibilities.

            rho[...] = b - np.dot(A, x) - w
            normr = np.sqrt(np.dot(rho, rho))

            sigma[...] = c - np.dot(At, y) + z
            norms = np.sqrt(np.dot(sigma, sigma))

            # STEP 2: Compute duality gap.

            gamma = np.dot(z, x) + np.dot(w, y)

            # Print statistics.
            primal_obj = np.dot(c, x) + f
            dual_obj = np.dot(b, y) + f
            if verbose > 0:
                print("{:8d}   {:14.7e}  {:8.1e}    {:14.7e}  {:8.1e} ".format(
                  _iter, primal_obj, normr, dual_obj, norms,))

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

            mu = delta * gamma / (n+m)

            # STEP 4: Compute step directions.
            D[...] = z/x
            E[...] = w/y

            # Create B matrix

            B[:m, :m] = np.diag(-E)  # top left
            B[:m, m:] = A   # top right
            B[m:, :m] = At  # bottom left
            B[m:, m:] = np.diag(D)  # bottom right

            dx = sigma - z + mu/x
            dy = rho + w - mu/y
            dxy[:m] = dy
            dxy[m:] = dx
            dxy = np.linalg.solve(B, dxy)
            dy = dxy[:m]
            dx = dxy[m:]

            dz = mu/x - z - D*dx
            dw = mu/y - w - E*dy

            # STEP 5: Ratio test to find step length.

            theta = max(0.0, np.max(-dx/x), np.max(-dz/z),
                        np.max(-dy/y), np.max(-dw/w))
            theta = min(r/theta, 1.0)

            # STEP 6: Step to new point

            x = x + theta*dx
            z = z + theta*dz

            y = y + theta*dy
            w = w + theta*dw

        normr0 = normr
        norms0 = norms

        self.status[ilp] = status
        self.x[ilp, :] = x
