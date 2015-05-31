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

def smx(m, n, a, ka, ia, x, y):
    """y = sparse matrix (a,ka,ia) times x"""
    y[...] = 0.0
    for j in range(n):
        for k in range(ka[j], ka[j+1]):
            y[ia[k]] += a[k]*x[j]


def atnum(m, n, ka, ia, a, kat, iat, at):
    """ (kat,iat,at) = transpose of (ka,ia,a)"""
    iwork = np.zeros(m, dtype=np.int)

    for k in range(ka[n]):
        row = ia[k]
        iwork[row]+=1

    kat[0] = 0
    for i in range(m):
        kat[i+1] = kat[i] + iwork[i]
        iwork[i] = 0

    for j in range(n):
        for k in range(ka[j], ka[j+1]):
            row = ia[k]
            addr = kat[row] +iwork[row]
            iwork[row]+=1
            iat[addr] = j
            at[addr]  = a[k]

    del(iwork)
