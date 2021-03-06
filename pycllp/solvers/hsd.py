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
from . import BaseCSCSolver
from ..ldlt import LDLTFAC
from ..linalg import smx, atnum

EPS = 1.0e-12
MAX_ITER = 200

class HSDSolver(BaseCSCSolver):
    name = 'hsd'
    def __init__(self, fdtype=np.float64, idtype=np.int64):
        super(HSDSolver, self).__init__()
        self.fdtype = fdtype
        self.idtype = idtype

    def init(self, A, b, c, f=0.0):
        super(HSDSolver, self).init(A, b, c, f=f)

    def solve(self, verbose=0):
        n,nlp = self.n,self.nlp
        self.x = np.empty((nlp, n), dtype=self.fdtype)
        self.status = np.empty(nlp, dtype=self.idtype)

        for i in range(nlp):
            self._solve(i, verbose=verbose)

        return self.status

    def _solve(self, ilp, verbose=0):
        n = self.n
        m = self.m
        nz = len(self.Ai)
        A = self.A[ilp, :].astype(self.fdtype)
        iA = self.Ai.astype(self.idtype)
        kA = self.Ak.astype(self.idtype)
        b = self.b[ilp,:].astype(self.fdtype)
        c= self.c[ilp,:].astype(self.fdtype)
        f=self.f[ilp].astype(self.fdtype)

        phi, psi, dphi, dpsi = 0.0, 0.0, 0.0, 0.0
        normr, norms = 0.0, 0.0  # infeasibilites
        gamma, delta, mu, theta = 0.0, 0.0, 0.0, 0.0  # parameters
        v = verbose
        status = 5
        primal_obj, dual_obj = 0.0, 0.0

        # Allocate memory for arrays.

        # Step direction
        dx = np.zeros(n, dtype=self.fdtype)
        dw = np.zeros(m, dtype=self.fdtype)
        dy = np.zeros(m, dtype=self.fdtype)
        dz = np.zeros(n, dtype=self.fdtype)
        # infeasibilites
        rho = np.zeros(m, dtype=self.fdtype)
        sigma = np.zeros(n, dtype=self.fdtype)
        # Diagonal matrixes
        D = np.zeros(n, dtype=self.fdtype)
        E = np.zeros(m, dtype=self.fdtype)

        fx = np.zeros(n, dtype=self.fdtype)
        fy = np.zeros(m, dtype=self.fdtype)
        gx = np.zeros(n, dtype=self.fdtype)
        gy = np.zeros(m, dtype=self.fdtype)

        # arrays for A^T
        At = np.zeros(nz, dtype=self.fdtype)
        iAt = np.zeros(nz, dtype=self.idtype)
        kAt = np.zeros(m+1, dtype=self.idtype)


        #  Initialization.

        x = np.ones(n, dtype=self.fdtype)
        z = np.ones(n, dtype=self.fdtype)
        w = np.ones(m, dtype=self.fdtype)
        y = np.ones(m, dtype=self.fdtype)

        phi = 1.0
        psi = 1.0

        atnum(m, n, kA, iA, A, kAt, iAt, At)

        # 	Display Banner.
        if verbose > 0:
            print("m = {:d},n = {:d},nz = {:d}".format(m, n, nz))
            print(
        """--------------------------------------------------------------------------
                 |           Primal          |            Dual           |       |
          Iter   |  Obj Value       Infeas   |  Obj Value       Infeas   |  mu   |
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        """)

        # 	Iteration.
        ldltfac = LDLTFAC(n, m, kAt, iAt, At, kA, iA, A, v)

        for _iter in range(MAX_ITER):
            # STEP 1: Compute mu and centering parameter delta.
            mu = (np.dot(z, x) + np.dot(w, y) + phi * psi) / (n+m+1)

            if _iter % 2 == 0:
                delta = 0.0
            else:
                delta = 1.0

            # STEP 1: Compute primal and dual objective function values.

            primal_obj = np.dot(c, x)
            dual_obj = np.dot(b, y)

            # STEP 2: Check stopping rule.

            if mu < EPS:
                if phi > psi:
                    status = 0
                    break  # OPTIMAL
                elif dual_obj < 0.0:
                    status = 2
                    break  # PRIMAL INFEASIBLE
                elif primal_obj > 0.0:
                    status = 4
                    break  # DUAL INFEASIBLE
                else:
                    print("Trouble in river city")
                    status = 4
                    break

            # STEP 3: Compute infeasibilities.
            smx(m, n, A, kA, iA, x, rho)
            for i in range(m):
                rho[i] = rho[i] - b[i]*phi + w[i]
            normr = np.sqrt(np.dot(rho, rho)) / phi

            for i in range(m):
                rho[i] = -(1-delta)*rho[i] + w[i] - delta*mu/y[i]

            smx(n, m, At, kAt, iAt, y, sigma)

            for j in range(n):
                sigma[j] = -sigma[j] + c[j]*phi + z[j]

            norms = np.sqrt(np.dot(sigma, sigma)) / phi

            for j in range(n):
                sigma[j] = -(1-delta)*sigma[j] + z[j] - delta*mu/x[j]

            gamma = -(1-delta)*(dual_obj - primal_obj + psi) + psi - delta*mu/phi

            # Print statistics.
            if verbose > 0:
                print("{:8d}   {:14.7e}  {:8.1e}    {:14.7e}  {:8.1e}  {:8.1e}".format(
                  _iter, primal_obj/phi+f, normr, dual_obj/phi+f, norms, mu))

            # STEP 4: Compute step directions.
            D[...] = z/x
            E[...] = w/y

            ldltfac.inv_num(E, D)

            for j in range(n):
                fx[j] = -sigma[j]

            for i in range(m):
                fy[i] =  rho[i]

            ldltfac.forwardbackward(E, D, fy, fx)

            for j in range(n):
                gx[j] = -c[j]

            for i in range(m):
                gy[i] = -b[i]

            ldltfac.forwardbackward(E, D, gy, gx)

            dphi = (np.dot(c,fx)-np.dot(b,fy)+gamma)/\
                    (np.dot(c,gx)-np.dot(b,gy)-psi/phi)

            for j in range(n):
                dx[j] = fx[j] - gx[j]*dphi
            for i in range(m):
                dy[i] = fy[i] - gy[i]*dphi

            for j in range(n):
                dz[j] = delta*mu/x[j] - z[j] - D[j]*dx[j]
            for i in range(m):
                dw[i] = delta*mu/y[i] - w[i] - E[i]*dy[i]
            dpsi = delta*mu/phi - psi - (psi/phi)*dphi

        	# STEP 5: Compute step length.

            if (_iter%2 == 0):
                pass
            else:
                theta = 1.0

            theta = 0.0
            for j in range(n):
                if (theta < -dx[j]/x[j]):
                    theta = -dx[j]/x[j]
                if (theta < -dz[j]/z[j]):
                    theta = -dz[j]/z[j]

            for i in range(m):
                if (theta < -dy[i]/y[i]):
                    theta = -dy[i]/y[i]
                if (theta < -dw[i]/w[i]):
                    theta = -dw[i]/w[i]

            if (theta < -dphi/phi):
                theta = -dphi/phi
            if (theta < -dpsi/psi):
                theta = -dpsi/psi
            try:
                theta = min( 0.95/theta, 1.0 )
            except ZeroDivisionError:
                theta = 1.0

        	# STEP 6: Step to new point

            for j in range(n):
                x[j] = x[j] + theta*dx[j]
                z[j] = z[j] + theta*dz[j]

            for i in range(m):
                y[i] = y[i] + theta*dy[i]
                w[i] = w[i] + theta*dw[i]

            phi = phi + theta*dphi
            psi = psi + theta*dpsi


        for j in range(n):
            x[j] /= phi
            z[j] /= phi

        for i in range(m):
            y[i] /= phi
            w[i] /= phi


        # 	Free work space                                             *


        del(     w )
        del(     z )
        del(    dx )
        del(    dw )
        del(    dy )
        del(    dz )
        del(   rho )
        del( sigma )
        del(     D )
        del(     E )
        del(    fx )
        del(    fy )
        del(    gx )
        del(    gy )

        del(   At )
        del(  iAt )
        del(  kAt )

        self.status[ilp] = status
        self.x[ilp,:] = x
