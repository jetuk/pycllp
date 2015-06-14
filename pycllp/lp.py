"""
Original implementation:
    Copyright (c) Robert J. Vanderbei, 1994
    All Rights Reserved
    http://www.princeton.edu/~rvdb/LPbook/

Port to Python
    James E. Tomlinson
    2015
"""
import sys
import numpy as np
from scipy.sparse import vstack, coo_matrix

class BaseLP(object):
    def __init__(self, A, b, c, f=0.0):
        self.A = A.tocsc().sorted_indices()
        self.b = b
        if self.b.ndim == 1:
            self.b = np.reshape(b, (1, len(b)))

        self.c = c
        if self.c.ndim == 1:
            self.c = np.reshape(c, (1, len(c)))
        if np.isscalar(f):
            self.f = np.ones(self.c.shape[0])*f
        else:
            self.f = f

    @property
    def m(self,):
        """Number of rows (constraints)"""
        return self.A.shape[0]


    @property
    def n(self,):
        """Number of columns (variables)"""
        return self.A.shape[1]

class StandardLP(BaseLP):

    def __init__(self, A, b, c, f=0.0):
        """
        Intialise with following general form,

        maximize:
        .. math:
            c^Tx

        subject to:
        .. math:
            Ax <= b
            x >= 0

        :param A: scipy.sparse matrix (will be converted to CSC,
            internally). Defines constraint coefficients.
        :param b: constraint upper bounds
        :param c: objective function coefficients
        """
        BaseLP.__init__(self, A, b, c, f=f)

    def init(self, solver):
        solver.init(self.A, self.b, self.c, self.f)


    def solve(self, solver, verbose=0):
        return solver.solve(verbose=verbose)


class GeneralLP(BaseLP):

    def __init__(self, A, b, c, r, l, u, f=0.0):
        """
        Intialise with following general form,

        optimize:
        .. math:
            c^Tx + f

        subject to:
        .. math:
            b <= Ax <= b+r
            l <=  x <= u

        :param A: scipy.sparse matrix (will be converted to CSC,
            internally). Defines constraint coefficients.
        :param b: constraint lower bounds
        :param c: objective function coefficients
        :param r: constraint range
        :param l: variable lower bounds
        :param u: variable upper bounds
        """
        BaseLP.__init__(self, A, b, c, f=f)
        self.r = r
        if self.r.ndim == 1:
            self.r = np.reshape(r, (1, len(r)))

        self.l = l
        if self.l.ndim == 1:
            self.l = np.reshape(l, (1, len(l)))

        self.u = u
        if self.u.ndim == 1:
            self.u = np.reshape(u, (1, len(u)))

    def to_standard_form(self,):
        """
        Return an instance of StandardLP by factoring this problem.
        """
        A = self.A.tocsc(copy=True)
        b = self.b.copy()
        c = self.c.copy()
        r = self.r.copy()
        l = self.l.copy()
        u = self.u.copy()
        f = self.f

        # abort if lower bound equals -Infinity
        if np.isneginf(self.l).any():
            raise ValueError('Lower bounds (l) contains -inf.')


        # shift lower bounds to zero (x <- x-l) so that new problem
        #  has the following form
        #
        #     optimize c^Tx + c^Tl
        #
        #     s.t. b-Al <= Ax <= b-Al+r
        #             0 <=  x <= u-l

        # indices where u is not +inf
        ind = np.where(np.isposinf(u)==False)[0]
        u[ind] -= l[ind]

        b = b - np.squeeze(A.dot(l.T))
        f = f + np.squeeze(np.dot(c, l.T))

        # Convert equality constraints to a pair of inequalities
        A = vstack([-A, A])  # Double A matrix

        b = np.c_[b, b]
        b[:,:self.m] *= -1
        b[:,self.m:] += r

        # add upper bounds
        nubs = len(ind)
        if nubs > 0:
            Aubs = coo_matrix((np.ones(nubs), (ind, ind)))
            b = np.r_[b,u[ind]]
            A = vstack([A,Aubs])

        #  Now lp has the following form,
        #
        #  maximize c^Tx + c^Tl
        #
        # s.t. -Ax <= -b
        #       Ax <=  b+r-l
        #        x <=  u-l
        #        x >=  0

        assert A.shape[0] == b.shape[1]

        lp = StandardLP(A,b,c,f=f)

        return lp
