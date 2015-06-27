import numpy as np
from scipy.sparse import rand, csc_matrix
from pycllp._ipo import ldltfac as c_ldltfac, getAAt
from pycllp.ldlt import LDLTFAC


def test_atnum():
    from pycllp.linalg import atnum
    m = 10
    n = 10
    A = rand(m, n, density=0.1, format='csc')

    At = np.zeros(A.nnz)
    iAt = np.zeros(A.nnz, dtype=np.int)
    kAt = np.zeros(m+1, dtype=np.int)

    atnum(m, n,  A.indptr, A.indices, A.data,
            kAt, iAt, At)

    scipy_At = A.transpose().tocsc()
    np.testing.assert_equal(scipy_At.indices, iAt)
    np.testing.assert_equal(scipy_At.indptr, kAt)
    np.testing.assert_almost_equal(scipy_At.data, At)

def test_ldlt():
    m = 10
    n = 10
    A = rand(m, n, density=0.1, format='csc')
    At = A.transpose().tocsc()
    dn = np.ones(n+m)
    dm = np.ones(m+n)
    print A.indptr, A.indices
    print At.indptr, At.indices
    print "Running ldltfac"
    c_ldltfac(m, n, A.indptr, A.indices, A.data, dn, dm,
              At.indptr, At.indices, At.data, 3)
    print "Getting AAt"
    ckAAt, ciAAt, cAAt = getAAt()

    ldltfac = LDLTFAC(m, n, A.indptr, A.indices, A.data,
                      At.indptr, At.indices, At.data, 3)
    ldltfac.inv_num(dn, dm)

    np.testing.assert_equal(ckAAt, ldltfac.kAAt)
    np.testing.assert_equal(ciAAt, ldltfac.iAAt)
    np.testing.assert_almost_equal(cAAt, ldltfac.AAt)
