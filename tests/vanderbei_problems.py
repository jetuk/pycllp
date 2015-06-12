
import numpy as np


def vanderbei_2_9():
    from scipy.sparse import csc_matrix
    A = np.array([ 1, 1, -2, 1, 2, -3, 2, 3], dtype=np.float)
    iA = np.array([1, 2,  0, 1, 2,  0, 1, 2], )
    kA = np.array([0, 2, 5, 8],)
    A = csc_matrix((A, iA, kA))

    b = np.array([-5, 0, 0], dtype=np.float)
    r = np.array([0, 4, 7], dtype=np.float)

    c = np.array([2, 3, 4], dtype=np.float)

    l = np.array([0, 0, 0], dtype=np.float)
    u = np.array([np.inf, np.inf, np.inf], dtype=np.float)
    f = 0.0

    xopt = np.array([1.5, 2.5, 0], dtype=np.float)

    return A, b, c, r, l, u, f, xopt
