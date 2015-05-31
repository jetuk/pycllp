

from _ipo import hsd_solver as solver


if __name__ == "__main__":
    import numpy as np

    # Example in Chapter 10 of Winston
    c = np.array([2.0,3,4])     # coefficients of variables in objective function.
                         # Program will automatically add subvector
    A = np.array(
        [[3.,2,1],   # Coefficient matrix,
        [2,5,3]])   # Only for "<" constraint equations!

    b = np.array([10., 15])       # Right hand side vector.

    m, n = A.shape

    A = A.T.flatten()
    nz = len(A)
    iA = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    kA = np.array([0, 2, 4, 6], dtype=np.int32)
    f = 0.001
    print(m, n, A, iA, kA)

    x = np.empty(n)
    y = np.empty(n)
    w = np.empty(n)
    z = np.empty(n)

    status = solver(m, n, nz, iA, kA, A, b, c, f, x, y, w, z)
    print(status, x, y)
