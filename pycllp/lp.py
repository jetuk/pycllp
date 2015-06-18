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


class SparseMatrix(object):
    """
    Sparse matrix data structure used by this package. It contains a single
    sparse structure, based on i,j coordinate arrays, and multi-dimensional
    data. Therefore many problems with the same structure but different values
    of A can be stored here.

    The structure is similar to the coo_matrix from scipy.sparse
    """
    def __init__(self, rows=None, cols=None, data=None, matrix=None):
        """
        SparseMatrix can be initiliased in two different ways,
        :param rows: array_like of i coordinates
        :param cols: array_like of j coordinates
        :param data: array_like of nonzero elements in A

        Or with a scipy sparse matrix
        :param matrix: scipy sparse matrix
        """
        if matrix is not None:
            # Create coordinate structure from data
            tmp = matrix.tocoo()
            self.rows = tmp.row
            self.cols = tmp.col
            # data reshaped so first dimension is the number of scenarios
            self.data = np.reshape(tmp.data, (1, len(tmp.data)))
        elif data is not None:
            if not len(rows) == len(cols) == len(data):
                raise ValueError("Arrays rows, cols and data must be the same length.")
            self.rows = np.array(rows)
            self.cols = np.array(cols)
            self.data = np.array(data)
            if self.data.ndim == 1: # Ensure data is 2-dimensional
                self.data = np.reshape(self.data, (1, len(self.data)))
        else:
            # Setup empty matrix
            self.rows = np.array([])
            self.cols = np.array([])
            self.data = np.array([[]])

    @property
    def nrows(self, ):
        try:
            return self.rows.max() + 1
        except ValueError:
            return 0

    @property
    def ncols(self, ):
        try:
            return self.cols.max() + 1
        except ValueError:
            return 0

    @property
    def nnzeros(self, ):
        return len(self.rows)

    @property
    def nproblems(self, ):
        return self.data.shape[0]

    def set_value(self, row, col, value):
        """
        Set value(s) to the sparse matrix

        :param rowi: row coordinate of entry
        :param col: col coordinate of entry
        :param value: value to apply to all problems. If scalar the same value
            is stored in all problems. Otherwise must be an array_like with same
            length as number of problems.
        """
        if row < 0 or col < 0:
            raise ValueError("Coordinates (i,j) must be >= 0")
        if not np.isscalar(value):
            if len(value) != self.data.shape[0]:
                raise ValueError("The number of coordinate values must match the number of problems.")

        # Check for existing entry
        ind = (self.rows==row) & (self.cols==col)
        if ind.sum() == 1:
            # existing entry; overwrite data
            self.data[:,ind] = value
        elif ind.sum() == 0:
            # new entry
            self.rows.resize(self.rows.shape[0]+1)
            self.cols.resize(self.cols.shape[0]+1)
            self.data.resize((self.data.shape[0],self.data.shape[1]+1))
            self.rows[-1] = row
            self.cols[-1] = col
            self.data[:,-1] = value
        else:
            raise ValueError("Multiple entries with the same coordinate pair. Bad things have happened!")

    def _del_value(self, row, col):
        """
        Delete value(s) from the sparse matrix. Use with caution as it can
        result it removing entire rows or columns which may create poorly
        formed matrices.

        :param row: row coordinate of entry
        :param col: col coordinate of entry
        """
        ind = (self.rows == row) & (self.cols == col)
        if ind.sum() == 1:
            # single entry to remove
            self.rows = self.rows[np.logical_not(ind)]
            self.cols = self.cols[np.logical_not(ind)]
            self.data = self.data[:, np.logical_not(ind)]
        else:
            raise ValueError("Multiple entries with the same coordinate pair. Bad things have happened!")

    def add_row(self, cols, value):
        """
        Add a row to the matrix.

        :param row: Coordinate of the row
        :param cols: iterable of column indices
        :param values: array_like either scalar, 1D or 2D. If 1D must be same length as
            cols, if the 2D shape is (problems, len(cols)).
        """
        # next row index is the current size of the matrix
        row = self.nrows
        v = np.array(value)
        for j, col in enumerate(cols):
            if np.isscalar(v):
                self.set_value(row, col, v)
            elif v.ndim == 1:
                self.set_value(row, col, v[j])
            elif v.ndim == 2:
                self.set_value(row, col, v[:, j])
            else:
                raise ValueError("Inconsistent data array provided.")
        return row

    def _del_row(self, row):
        """
        Delete a row from the matrix. Use with caution as it can result in gaps
        in the matrix.
        """
        # find all coordinates with the row
        ind = (self.rows == row)
        self.rows = self.rows[np.logical_not(ind)]
        self.cols = self.cols[np.logical_not(ind)]
        self.data = self.data[:, np.logical_not(ind)]

    def update_row(self, row, cols, value):
        """
        Update a row in the matrix. If the row does not exist it will be added.
        Existing entries for the row will be removed first.
        """
        # Remove the row first.
        self._del_row(row)
        # Now add new data
        v = np.array(value)
        for j, col in enumerate(cols):
            if np.isscalar(v):
                self.set_value(row, col, v)
            elif v.ndim == 1:
                self.set_value(row, col, v[j])
            elif v.ndim == 2:
                self.set_value(row, col, v[:, j])
            else:
                raise ValueError("Inconsistent data array provided.")

    def add_col(self, rows, value):
        """
        Add a column to the matrix.

        :param col: Coordinate of the column
        :param rows: iterable of row indices
        :param values: array_like either scalar, 1D or 2D. If 1D must be same length as
            cols, if the 2D shape is (problems, len(cols)).
        """
        # next column index is the current size of the matrix
        col = self.ncols
        v = np.array(value)
        for i, row in enumerate(rows):
            if np.isscalar(v):
                self.set_value(row, col, v)
            elif v.ndim == 1:
                self.set_value(row, col, v[i])
            elif v.ndim == 2:
                self.set_value(row, col, v[:, i])
            else:
                raise ValueError("Inconsistent data array provided.")
        return col

    def _del_col(self, col):
        """
        Delete a column from the matrix. Use with caution as it can result in
        gaps in the matrix.
        """
        # find all coordinates with the row
        ind = (self.cols == col)
        self.rows = self.rows[np.logical_not(ind)]
        self.cols = self.cols[np.logical_not(ind)]
        self.data = self.data[:, np.logical_not(ind)]

    def update_col(self, col, rows, value):
        """
        Update a column in the matrix. If the row does not exist it will be added.
        Existing entries for the row will be removed first.
        """
        # Remove the row first.
        self._del_row(row)
        # Now add new data
        v = np.array(value)
        for i, row in enumerate(rows):
            if np.isscalar(v):
                self.set_value(row, col, v)
            elif v.ndim == 1:
                self.set_value(row, col, v[i])
            elif v.ndim == 2:
                self.set_value(row, col, v[:, i])
            else:
                raise ValueError("Inconsistent data array provided.")

    def tocoo(self, problem=0):
        return coo_matrix( (self.data[problem,:], (self.rows, self.cols)) )

    def tocsc(self, problem=0):
        return self.tocoo().tocsc()

    def todense(self, problem=0):
        return self.tocoo().todense()



class StandardLP(object):
    """
    Base class for LP models.

    A matrix is stored as a list of coordinates.
    """
    def __init__(self, A=None, b=None, c=None, f=None):
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
        if A is not None:
            if b is None or c is None or f is None:
                raise ValueError("If A matrix is provided then b, c and f must also be provided")

            self.A = A
            nprb = self.A.nproblems

            self.b = np.array(b)
            if self.b.ndim == 1:
                self.b =  np.array(np.dot(np.ones((nprb,1)),np.matrix(self.b)))

            self.c = np.array(c)
            if self.c.ndim == 1:
                self.c =  np.array(np.dot(np.ones((nprb,1)),np.matrix(self.c)))

            if np.isscalar(f):
                self.f = np.ones(nprb)*f
            else:
                self.f = np.array(f)
        else:
            # A not provided, create empty arrays
            self.A = SparseMatrix()
            self.b = np.array([[]])
            self.c = np.array([[]])
            self.f = np.array([])

    @property
    def nrows(self, ):
        return self.A.nrows

    @property
    def ncols(self, ):
        return self.A.ncols

    @property
    def nnzeros(self, ):
        return self.A.nnzeros

    @property
    def nproblems(self, ):
        return self.A.nproblems

    @property
    def m(self,):
        """Number of rows (constraints)"""
        return self.nrows

    @property
    def n(self,):
        """Number of columns (variables)"""
        return self.ncols

    def init(self, solver):
        solver.init(self.A, self.b, self.c, self.f)

    def solve(self, solver, verbose=0):
        return solver.solve(verbose=verbose)


class GeneralLP(StandardLP):

    def __init__(self, Ai, Aj, Adata, b, c, r, l, u, f=0.0):
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
        StandardLP.__init__(self, Ai, Aj, Adata, b, c, f=f)
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
