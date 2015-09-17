from pycllp.lp import SparseMatrix, StandardLP, GeneralLP
import numpy as np
from numpy.testing import assert_allclose
import pytest

class TestScalarMatrix(object):

    def test_empty_init(self, ):
        A = SparseMatrix()

        assert A.nrows == 0
        assert A.ncols == 0
        assert A.nnzeros == 0
        assert A.nproblems == 1

    def test_matrix_init(self, ):
        from scipy.sparse import csc_matrix
        B = np.reshape(np.arange(1,7,dtype=np.float32),(3,2))
        A = SparseMatrix(matrix=csc_matrix(B))

        assert A.nrows == 3
        assert A.ncols == 2
        assert A.nnzeros == 6
        assert A.nproblems == 1

    def test_array_init(self, ):
        A = SparseMatrix([0,0,1,1,2,2],[0,1,0,1,0,1],np.arange(6))

        assert A.nrows == 3
        assert A.ncols == 2
        assert A.nnzeros == 6
        assert A.nproblems == 1

    def test_set_value(self, ):
        A = SparseMatrix()

        A.set_value(0, 0, 1.0)

        assert A.nrows == 1
        assert A.ncols == 1
        assert A.nnzeros == 1
        assert A.nproblems == 1

        A.set_value(0, 1, 1.0)

        assert A.nrows == 1
        assert A.ncols == 2
        assert A.nnzeros == 2
        assert A.nproblems == 1

        A._del_value(0, 0)

        assert A.nrows == 1
        # There are still two columns because deleting anything
        # but the last entry does not alter the size.
        assert A.ncols == 2
        assert A.nnzeros == 1
        assert A.nproblems == 1

        A._del_value(0, 1)

        assert A.nrows == 0
        # Now now columns or rows as there are no values in matrix
        assert A.ncols == 0
        assert A.nnzeros == 0
        assert A.nproblems == 1

    def test_add_row(self, ):
        A = SparseMatrix()

        row = A.add_row([0,2,3], [1.0,1.0,1.0])
        assert row == 0
        assert A.nrows == 1
        assert A.ncols == 4
        assert A.nnzeros == 3
        assert A.nproblems == 1

        row = A.add_row([1], [1.0])
        assert row == 1
        assert A.nrows == 2
        assert A.ncols == 4
        assert A.nnzeros == 4
        assert A.nproblems == 1

        A._del_row(1)

        assert A.nrows == 1
        assert A.ncols == 4
        assert A.nnzeros == 3
        assert A.nproblems == 1

        A._del_row(0)

        assert A.nrows == 0
        assert A.ncols == 0
        assert A.nnzeros == 0
        assert A.nproblems == 1

    def test_add_col(self, ):
        A = SparseMatrix()

        col = A.add_col([0,2,3], [1.0,1.0,1.0])
        assert col == 0
        assert A.nrows == 4
        assert A.ncols == 1
        assert A.nnzeros == 3
        assert A.nproblems == 1

        col = A.add_col([1], [1.0])
        assert col == 1
        assert A.nrows == 4
        assert A.ncols == 2
        assert A.nnzeros == 4
        assert A.nproblems == 1

        A._del_col(1)

        assert A.nrows == 4
        assert A.ncols == 1
        assert A.nnzeros == 3
        assert A.nproblems == 1

        A._del_col(0)

        assert A.nrows == 0
        assert A.ncols == 0
        assert A.nnzeros == 0
        assert A.nproblems == 1


class TestStandardLP(object):

    def test_empty_init(self, ):
        lp = StandardLP()

        assert lp.nrows == 0
        assert lp.ncols == 0
        assert lp.nnzeros == 0
        assert lp.nproblems == 1

    def test_array_init(self, ):
        from scipy.sparse import csc_matrix
        B = np.reshape(np.arange(1,7,dtype=np.float32),(3,2))
        A = SparseMatrix(matrix=csc_matrix(B))
        b = np.zeros(A.nrows)
        c = np.zeros(A.ncols)
        f = 0.0

        lp = StandardLP(A, b, c, f)

        assert lp.nrows == 3
        assert lp.ncols == 2
        assert lp.nnzeros == 6
        assert lp.nproblems == 1

    def test_add_row(self, ):
        lp = StandardLP()

        row = lp.add_row([0,2,3], [1.0,1.0,1.0], 1.0)

        assert row == 0
        assert lp.nrows == 1
        assert lp.ncols == 4
        assert lp.nnzeros == 3
        assert lp.nproblems == 1
        assert_allclose(lp.b, [[1.0]])

        lp.set_bound(0, 2.0)
        assert_allclose(lp.b, [[2.0]])

        with pytest.raises(ValueError):
            lp.set_bound(1, 1.0)

        cols, value, bound = lp.get_row(0)
        assert_allclose(cols, [0,2,3])
        assert_allclose(value, [[1.0,1.0,1.0]])
        assert bound == 2.0

    def test_add_col(self, ):
        lp = StandardLP()

        col = lp.add_col([0,2,3], [1.0,1.0,1.0], 1.0)

        assert col == 0
        assert lp.nrows == 4
        assert lp.ncols == 1
        assert lp.nnzeros == 3
        assert lp.nproblems == 1
        assert_allclose(lp.c, [[1.0]])

        lp.set_objective(0, 2.0)
        assert_allclose(lp.c, [[2.0]])

        with pytest.raises(ValueError):
            lp.set_objective(1, 1.0)


class TestGeneralLP(object):

    def test_empty_init(self, ):
        lp = GeneralLP()

        assert lp.nrows == 0
        assert lp.ncols == 0
        assert lp.nnzeros == 0
        assert lp.nproblems == 1

    def test_array_init(self, ):
        from scipy.sparse import csc_matrix
        B = np.reshape(np.arange(1, 7, dtype=np.float32), (3, 2))
        A = SparseMatrix(matrix=csc_matrix(B))
        b = np.zeros(A.nrows)
        c = np.zeros(A.ncols)
        f = 0.0

        lp = GeneralLP(A, b, c, f=f)

        assert lp.nrows == 3
        assert lp.ncols == 2
        assert lp.nnzeros == 6
        assert lp.nproblems == 1

    def test_gte_conversion(self, ):
        lp = GeneralLP()
        cols = [0, 1, 2]
        vals = [[1., 1., 1.]]
        lp.add_row(cols, vals, 2.0, np.inf)

        slp = lp.to_standard_form()

        slp_cols, slp_vals, slp_bound = slp.get_row(0)

        assert_allclose(cols, slp_cols)
        assert_allclose(vals, -slp_vals)
        assert_allclose([2.0], -slp_bound)

        with pytest.raises(IndexError):
            slp_cols, slp_vals, slp_bound = slp.get_row(1)



class TestVanderbei2_9(object):

    def test_vanderbei_2_9(self, ):
        from vanderbei_problems import vanderbei_2_9
        args = list(vanderbei_2_9())
        args.pop()  # remove xopt
        lp = GeneralLP(*args)
        # Test structure
        lp_cols, lp_vals, lp_lb, lp_ub = lp.get_row(0)
        assert_allclose([1, 2], lp_cols)
        assert_allclose([[-2, -3]], lp_vals)
        assert_allclose([-5], lp_lb)
        assert_allclose([np.inf], lp_ub)

        slp = lp.to_standard_form()

        slp_cols, slp_vals, slp_bound = slp.get_row(0)
