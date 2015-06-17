from pycllp.lp import SparseMatrix, StandardLP
import numpy as np

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


class TestBaseLP(object):

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

        assert A.nrows == 3
        assert A.ncols == 2
        assert A.nnzeros == 6
        assert A.nproblems == 1
