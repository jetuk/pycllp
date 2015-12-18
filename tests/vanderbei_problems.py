
import numpy as np
from pycllp.lp import SparseMatrix, EqualityLP, StandardLP


def vanderbei_2_9():
    from scipy.sparse import csc_matrix
    A = np.array([ 1, 1, 2, 1, 2, 3, 2, 3], dtype=np.float)
    iA = np.array([1, 2, 0, 1, 2,  0, 1, 2], )
    kA = np.array([0, 2, 5, 8],)
    A = csc_matrix((A, iA, kA)).tocoo()

    b = np.array([5, 4, 7], dtype=np.float)
    c = np.array([2, 3, 4], dtype=np.float)

    xopt = np.array([1.5, 2.5, 0], dtype=np.float)

    lp = StandardLP(SparseMatrix(matrix=A), b, c, 0.0)

    return lp, xopt


def vanderbei_2_10():
    from scipy.sparse import csc_matrix
    A = np.array([ 1, 1, 1, 1], dtype=np.float)
    iA = np.array([0, 0, 0, 0], )
    kA = np.array([0, 1, 2, 3, 4],)
    A = csc_matrix((A, iA, kA))

    b = np.array([1, ], dtype=np.float)
    c = np.array([6, 8, 5, 9], dtype=np.float)

    xopt = np.array([0, 0, 0, 1], dtype=np.float)

    lp = EqualityLP(SparseMatrix(matrix=A), b, c, 0.0)

    return lp, xopt
