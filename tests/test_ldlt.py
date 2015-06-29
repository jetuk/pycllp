import numpy as np
from scipy.sparse import rand, csc_matrix
from pycllp._ipo import ldltfac as c_ldltfac, getAAt, inv_clo
from pycllp.ldlt import LDLTFAC
import pytest


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


@pytest.mark.parametrize("N", [10, 20, 50, 100, 250])
def test_ldlt(N):
    m = n = N
    np.random.seed(0)
    A = rand(m, n, density=0.1, format='csc')
    At = A.transpose().tocsc()
    dn = np.ones(n+m)
    dm = np.ones(m+n)

    c_ldltfac(m, n, A.indptr, A.indices, A.data, dn, dm,
              At.indptr, At.indices, At.data, 0)

    ckAAt, ciAAt, cAAt = getAAt()

    ldltfac = LDLTFAC(m, n, A.indptr, A.indices, A.data,
                      At.indptr, At.indices, At.data, 0)
    ldltfac.inv_num(dn, dm)

    np.testing.assert_equal(ckAAt, ldltfac.kAAt)
    np.testing.assert_equal(ciAAt, ldltfac.iAAt)
    np.testing.assert_almost_equal(cAAt, ldltfac.AAt)

    inv_clo()
