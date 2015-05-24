"""
Original implementation:
    Copyright (c) Robert J. Vanderbei, 1994
    All Rights Reserved
    http://www.princeton.edu/~rvdb/LPbook/

Port to Python
    James E. Tomlinson
    2015
"""
import numpy as np
from lp import LP
from linalg import smx
import sys

HUGE_VAL = sys.float_info.max


class LDLTFAC(object):
    _EPS = 1.0e-8
    _EPSSOL = 1.0e-6  # Zero tolerance for consistent eqns w/dep rows
    _EPSNUM = 0.0  # Initial zero tolerance for dependent rows
    _EPSCDN = 1.0e-12  # Zero tolerance for ill-conditioning test
    _EPSDIAG = 1.0e-14  # diagonal perturbation
    _STABLTY = 1.0  # 0=fast/unstable, 1=slow/stable
    _NOREORD = 0
    _MD = 1
    _MLF = 2
    _DENSE = -1
    _UNSET = 0
    _PRIMAL = 1
    _DUAL = 2

    def __init__(self,
                m, # rows
                n, # columns
                kA,  # constraint matrix in three linear arrays
                iA,
                A,
                kAt,  # A^T in three linear arrays
                iAt,
                At,
                verbose  # verbosity
                ):
        self.kQ = np.zeros(n+1, dtype=np.int)
        self.bndmark = np.zeros(n, dtype=np.int)
        self.rngmark = np.zeros(m, dtype=np.int)

        self.bndmark[...] = LP.BDD_BELOW
        self.rngmark[...] = LP.INFINITE

        self.m = m
        self.n = n
        self.kA = kA
        self.iA = iA
        self.A  =  A
        self.kAt = kAt
        self.iAt = iAt
        self.At  =  At
        self.verbose = verbose

        self.diag = None
        self.iQ = None
        self.Q = None
        self.tier = None
        self.max = 0
        self.mark = None
        self.ndep = None

        self.perm, self.iperm, self.iAAt, self.kAAt = None, None, None, None
        self.mark, self.AAt, self.diag = None, None, None

        self.epssol = self._EPSSOL
        self.epsnum = self._EPSNUM
        self.epscdn = self._EPSCDN
        self.epsdiag= self._EPSDIAG
        self.stablty= self._STABLTY
        self.method = self._MD
        self.dense  = self._DENSE
        self.pdf    = self._UNSET

        self.y_k = None
        self.x_k = None
        self.r = None
        self.s = None
        self.z = None
        self.Qx = None

        #self.inv_num(dn, dm)

    def inv_num(self,
                dn,  # diagonal matrix for upper-left  corner
                dm  # diagonal matrix for lower-right corner
                ):
        m       =self.m
        n       =self.n
        kA     =self.kA
        iA     =self.iA
        A      =self.A
        kAt    =self.kAt
        iAt    =self.iAt
        At     =self.At
        kQ     =self.kQ
        iQ     =self.iQ
        Q      =self.Q

        #/*----------------------------------------------+
        #| Set up                                        |
        #|            /       2      t \                 |
        #|            | -( Q+D  )   A  |                 |
        #|     K  =   |       n        |                 |
        #|            |              2 |                 |
        #|            |     A       D  |                 |
        #|            \              m /                 |
        #|                                               |
        #| in AAt, iAAt, kAAt, and diag. Only the lower  |
        #| triangle of a permutation of K is stored.     |
        #|                                              */

        #/*----------------------------------------------+
        #| If data structures are not set up, read       |
        #| $HOME/.syseq to set parameters and then       |
        #| do a symbolic factorization.  The symmetric   |
        #| reordering is put into perm[] and its         |
        #| inverse is put into iperm[]:                  |
        #|      new_index = iperm[ old_index ]           |
        #|                                              */

        if self.diag is None:
            self.inv_sym()

        diag = self.diag
        perm = self.perm
        iperm = self.iperm
        kAAt = self.kAAt
        iAAt = self.iAAt
        AAt = self.AAt
        #/*----------------------------------------------+
        #| Get memory for integer work array.            |
        #|                                              */

        iwork = np.zeros(m+n, dtype=np.int)

        #/*----------------------------------------------+
        #| Store the diagonal of K in diag[].            |
        #|                                              */

        for j in range(n):
            diag[iperm[j]] = -max(dn[j], self.epsdiag)
        for i in range(m):
            diag[iperm[n+i]] = max(dm[i], self.epsdiag)

        #/*----------------------------------------------+
        #| Store lower triangle of permutation of K      |
        #| in AAt[], iAAt[], kAAt[].                     |
        #|                                              */

        for j in range(n):
            col = iperm[j]  # col is a new_index
            for k in range(kAAt[col], kAAt[col+1]):
                iwork[iAAt[k]] = k
                AAt[k] = 0.0

            for k in range(kA[j], kA[j+1]):
                row = iperm[n+iA[k]]  # row is a new_index
                if row > col:
                    AAt[iwork[row]] = A[k]

            for k in range(kQ[j], kQ[j+1]):
                row = iperm[iQ[k]]  # row is a new_index
                if row > col:
                    AAt[iwork[row]] = -max*Q[k]
                elif row == col:
                    diag[row] -= max*Q[k]

        for i in range(m):
            col = iperm[n+i]
            for k in range(kAAt[col], kAAt[col+1]):
                iwork[iAAt[k]] = k
                AAt[k] = 0.0

            for k in range(kAt[i], kAt[i+1]):
                row = iperm[iAt[k]]
                if row > col:
                    AAt[iwork[row]] = At[k]

        del(iwork)

        #/*----------------------------------------------+
        #| Going into lltnum, any row/column for which   |
        #| mark[] is set to FALSE will be treated as     |
        #| a non-existing row/column.  On return,        |
        #| any dependent rows are also marked FALSE.     |
        #|                                              */

        self.mark = np.ones(m+n, dtype=np.bool)

        self.lltnum()

        for i in range(m+n):
            if ((perm[i] < n and diag[i] > 0.0 and self.verbose > 2) or
    	         (perm[i] >= n and diag[i] < 0.0 and self.verbose > 2)):
                print("nonconvex subproblem: diag[{:4d}] = {:10.2e}".format(
    				         i, diag[i]))

        #double mindiag = HUGE_VAL

        #for (i=0 i<m+n i++) {
        #    if (ABS(diag[i]) < mindiag)
        #            mindiag = ABS(diag[i])
        #}

        #if ( mindiag < 1.0e-14 ) {
        #    epsdiag *= 10
        #    if (verbose>2) printf("mindiag = %10.2e, epsdiag = %10.2e\n",
        #                           high(mindiag), high(epsdiag))
        #}

        if self.verbose>1 and self.ndep>0:
            print("     dependent  rows:    {:d}".format(self.ndep))


    def forwardbackward(self,
                        Dn,    # diagonal matrix for upper-left  corner
                        Dm,    # diagonal matrix for lower-right corner
                        dx,
                        dy,):

        self.solve(Dn,Dm,dx,dy)


    def solve(self, Dn, Dm, c, b):
        """
        /*----------------------------------------------+
        | The routine solve() uses rawsolve() together  |
        | with iterative refinement to produce the best |
        | possible solutions for the given              |
        | factorization.                               */
        """
        m   =self.m
        n   =self.n
        kA =self.kA
        iA =self.iA
        A  =self.A
        kAt=self.kAt
        iAt=self.iAt
        At =self.At
        kQ =self.kQ
        iQ =self.iQ
        Q  =self.Q
        _max =self.max

        _pass=0
        consistent = True

        m2 = m+n

        # Replaced MALLOC & REALLOC with np.empty
        if self.y_k is None:
            self.y_k = np.empty(m)
        else:
            self.y_k.resize(m)
        if self.x_k is None:
            self.x_k = np.empty(n)
        else:
            self.x_k.resize(n)
        if self.r is None:
            self.r = np.empty(m)
        else:
            self.r.resize(m)
        if self.s is None:
            self.s = np.empty(n)
        else:
            self.s.resize(n)
        if self.z is None:
            self.z = np.empty(m2)
        else:
            self.z.resize(m2)
        if self.Qx is None:
            self.Qx = np.empty(n)
        else:
            self.Qx.resize(n)

        y_k = self.y_k
        x_k = self.x_k
        r = self.r
        s = self.s
        z = self.z
        Qx = self.Qx
        iperm = self.iperm

        maxbc = max(np.abs(b).max(), np.abs(c).max()) + 1
        maxrs = HUGE_VAL

        while True:

            if _pass == 0:
                for j in range(n):
                    z[iperm[j]]   = c[j]
                for i in range(m):
                    z[iperm[n+i]] = b[i]
            else:
                for j in range(n):
                    z[iperm[j]]   = s[j]
                for i in range(m):
                    z[iperm[n+i]] = r[i]

            consistent = self.rawsolve()

            if _pass == 0:
                for j in range(n):
                    x_k[j] = z[iperm[j]]
                for i in range(m):
                    y_k[i] = z[iperm[n+i]]
            else:
                for j in range(n):
                    x_k[j] = x_k[j] + z[iperm[j]]
                for i in range(m):
                    y_k[i] = y_k[i] + z[iperm[n+i]]

            smx(m, n, A, kA, iA, x_k,r)
            smx(n, m, At,kAt,iAt,y_k,s)
            smx(n, n, Q ,kQ ,iQ ,x_k,Qx)

            for j in range(n):
                s[j] = c[j] - (s[j] - Dn[j]*x_k[j] - _max*Qx[j])
            for i in range(m):
                r[i] = b[i] - (r[i] + Dm[i]*y_k[i])

            oldmaxrs = maxrs
            maxrs = max(np.abs(r).max(), np.abs(s).max())

            # --- for tuning purposes ---
            if (self.verbose>2 and _pass>0):
                print("refinement({:3d}): {:8.2e} {:8.2e} {:8.2e}".format(
                    _pass, np.abs(s).max(), np.abs(r).max(), maxrs/maxbc ))

            _pass += 1
            if maxrs > (1.0e-10*maxbc) and maxrs < (oldmaxrs/2):
                continue
            else:
                break

        if maxrs > oldmaxrs and _pass > 1:
            for j in range(n):
                x_k[j] = x_k[j] - z[iperm[j]]
            for i in range(m):
                y_k[i] = y_k[i] - z[iperm[n+i]]

    	#----------------------------------------------------------
    	#| overwrite input with output

        for j in range(n):
            c[j] = x_k[j]
        for i in range(m):
            b[i] = y_k[i]

        return consistent



    def rawsolve(self,):
        """
        /*----------------------------------------------+
        | The routine rawsolve() does the forward,      |
        | diagonal, and backward substitions to solve   |
        | systems of equations involving the known      |
        | factorization.                               */
        """
        m = self.m
        n = self.n
        z = self.z
        mark = self.mark
        kAAt = self.kAAt
        iAAt = self.iAAt
        AAt = self.AAt
        diag = self.diag
        consistent = True
        eps = 0.0
        m2 = m+n

        if self.ndep:
            eps = self.epssol * np.abs(z).max()

        #/*------------------------------------------------------+
        #|                                                       |
        #|               -1                                      |
        #|       z  <-  L  z                                     |
        #|                                                      */

        for i in range(m2):
            if mark[i]:
                beta = z[i]
                for k in range(kAAt[i], kAAt[i+1]):
                    row = iAAt[k]
                    z[row] -= AAt[k]*beta
            elif abs(z[i]) > eps:
                consistent = False
            else:
                z[i] = 0.0

        #/*------------------------------------------------------+
        #|                                                       |
        #|               -1                                      |
        #|       z  <-  D  z                                     |
        #|                                                      */

        for i in range(m2-1, -1, -1):
            if mark[i]:
                z[i] = z[i]/diag[i]
            elif abs(z[i]) > eps:
                consistent = False
            else:
                z[i] = 0.0

        #/*------------------------------------------------------+
        #|                                                       |
        #|                t -1                                   |
        #|       z  <-  (L )  z                                  |
        #|                                                      */

        for i in range(m2-1, -1, -1):
            if mark[i]:
                beta = z[i]
                for k in range(kAAt[i], kAAt[i+1]):
                    beta -= AAt[k]*z[iAAt[k]]
                z[i] = beta
            elif abs(z[i]) > eps:
                consistent = False
            else:
                z[i] = 0.0

        return consistent


    def inv_clo(self, ):
        self.perm, self.iperm, self.iAAt, self.kAAt = None, None, None, None
        self.mark, self.AAt, self.diag = None, None, None
        self.y_k, self.x_k = None, None
        self.r, self.s, self.z, self.Qx = None, None, None, None

    # Define static functions

    def lltnum(self,):
        """
        /*------------------------------------------------------+
        |                                                       |
        | the input is a symmetric matrix  A  with lower        |
        |       triangle stored sparsely in                     |
        |       kAAt[], iAAt[], AAt[] and with the diagonal     |
        |       stored in dia[].                                |
        | the output is the lower triangular matrix  L          |
        |       stored sparsely in  kAAt,iAAt,AAt  and          |
        |       a diagonal matrix  D  stored in the diag.       |
        |                  t                                    |
        |          A  = LDL                                     |
        |                                                       |
        +------------------------------------------------------*/
        """
        m = self.m
        n = self.n
        diag = self.diag
        perm = self.perm
        AAt = self.AAt
        kAAt = self.kAAt
        iAAt = self.iAAt
        mark = self.mark
        self.denwin

        m2 = m+n
        #/*------------------------------------------------------+
        #| initialize constants                                 */

        temp = np.empty(m2)
        first = np.empty(m2, dtype=np.int)
        link = np.empty(m2, dtype=np.int)
        for i in range(m2):
            link[i] = -1

        maxdiag=0.0
        for i in range(m2):
            if abs(diag[i]) > maxdiag:
                maxdiag = abs(diag[i])

        self.ndep=0

        #/*------------------------------------------------------+
        #| begin main loop - this code is taken from George and  |
        #| Liu's book, pg. 155, modified to do LDLt instead      |
        #| of LLt factorization.                                */

        for i in range(m2):
            diagi = diag[i]
            sgn_diagi = -1 if perm[i] < n else 1
            j = link[i]
            while j != -1:
                newj = link[j]
                k = first[j]
                lij = AAt[k]
                lij_dj = lij*diag[j]
                diagi -= lij*lij_dj
                k_bgn = k+1
                k_end = kAAt[j+1]
                if k_bgn < k_end:
                    first[j] = k_bgn
                    row = iAAt[k_bgn]
                    link[j] = link[row]
                    link[row] = j
                    if j < self.denwin:
                        for kk in range(k_bgn, k_end):
                            temp[iAAt[kk]] += lij_dj*AAt[kk]
                    else:
                        ptr = row
                        for kk in range(k_bgn, k_end):
                            temp[ptr] += lij_dj*AAt[kk]
                            ptr+=1

                j=newj

            k_bgn = kAAt[i]
            k_end = kAAt[i+1]
            for kk in range(k_bgn, k_end):
                row = iAAt[kk]
                AAt[kk] -= temp[row]

            if abs(diagi) <= self.epsnum*maxdiag or mark[i] == False:

            #if (sgn_diagi*diagi <= epsnum*maxdiag || mark[i] == FALSE)

                self.ndep+=1
                maxoffdiag = 0.0
                for kk in range(k_bgn, k_end):
                    maxoffdiag = max( maxoffdiag, abs( AAt[kk] ) )

                if maxoffdiag < 1.0e+6*self._EPS:
                    mark[i] = False
                else:
                    diagi = sgn_diagi * self._EPS

            diag[i] = diagi
            if k_bgn < k_end:
                first[i] = k_bgn
                row = iAAt[k_bgn]
                link[i] = link[row]
                link[row] = i
                for kk in range(k_bgn, k_end):
                    row = iAAt[kk]
                    if mark[i]:
                        AAt[kk] /= diagi
                    else:
                        AAt[kk] = 0.0

                    temp[row] = 0.0

        del(link)
        del(first)
        del(temp)

    def inv_sym(self, ):
        r"""
        /*----------------------------------------------+
        | Set up adjacency structure for                |
        |                                               |
        |            /       2      t \                 |
        |            | -( Q+D  )   A  |                 |
        |     K  =   |       n        |                 |
        |            |              2 |                 |
        |            |     A       D  |                 |
        |            \              m /                 |
        |                                               |
        +----------------------------------------------*/
        """
        m = self.m
        n = self.n
        kQ = self.kQ
        iQ = self.iQ
        iA = self.iA
        kA = self.kA
        kAt = self.kAt
        iAt = self.iAt
        bndmark = self.bndmark
        rngmark = self.rngmark

        verbose = self.verbose
        pdf = self.pdf

        separable = True
        degree = np.empty(n+m, dtype=np.int)
        nbrs = np.empty(n+m, dtype=object)

        #/*-----------------------------------------------------+
        #| First check to see if the problem is separable.     */

        for j in range(n):
            for k in range(kQ[j], kQ[j+1]):
                if iQ[k] != j:
                    separable = False
                    break

        #/*----------------------------------------------------+
        #| Select ordering priority (primal or dual)          */


        _dense, _fraction, pfillin, dfillin = 0.0, 0.0, 0.0, 0.0

        _fraction = 1.0e0
        for j in range(n):
            _dense = float(kA[j+1]-kA[j])/(m+1)
            _fraction = _fraction*(1.0e0 - _dense*_dense)

        pfillin = 0.5*m*m*(1.0e0-_fraction)
        if verbose>2:
            print("primal fillin estimate: {:10.0f}".format(pfillin))

        _fraction = 1.0e0
        for i in range(m):
            _dense = float(kAt[i+1]-kAt[i])/(n+1)
            _fraction = _fraction*(1.0e0 - _dense*_dense)

        dfillin = 0.5*n*n*(1.0e0-_fraction)
        if verbose>2:
            print("dual   fillin estimate: {:10.0f}\n".format(dfillin))

        if pdf == self._UNSET:
            if 3*pfillin <= dfillin and separable:
                pdf = self._PRIMAL
                if verbose>2:
                    print("Ordering priority favors PRIMAL")
            else:
                pdf = self._DUAL
                if verbose>2:
                    print("Ordering priority favors DUAL")


        #/*----------------------------------------------+
        #| Initialize nbrs so that nbrs[col][k] con-     |
        #| tains the row index of the k_th nonzero in    |
        #| column col.                                   |
        #| Initialize degree so that degree[col] con-    |
        #| tains the number of nonzeros in column col.   |
        #|                                              */

        for j in range(n):
            ne = kA[j+1] - kA[j] + kQ[j+1] - kQ[j]
            nbrs[j] = np.empty(ne, dtype=np.int)
            ne = 0
            for k in range(kA[j], kA[j+1]):
                nbrs[j][ne]    = n+iA[k]
                ne+=1
            for k in range(kQ[j],kQ[j+1]):
                if iQ[k] != j:
                    nbrs[j][ne]    = iQ[k]
                    ne+=1

            degree[j] = ne


        for i in range(m):
            ne = kAt[i+1] - kAt[i]
            nbrs[n+i] = np.empty(ne, dtype=np.int)
            degree[n+i] = ne
            ne = 0
            for k in range(kAt[i], kAt[i+1]):
                nbrs[n+i][ne]    = iAt[k]
                ne+=1

        #/*----------------------------------------------+
        #| Initialize tier to contain the ordering       |
        #| priority scheme.                              |
        #|                                              */

        if self.tier is None:
            self.tier = np.empty(n+m, dtype=np.int)
            n1 = 0
            if pdf == self._PRIMAL:
                for j in range(n):
                    if bndmark[j] != LP.FREEVAR:
                        self.tier[j] = 0  # 0
                    else:
                        self.tier[j] = 1  # 2

                for i in range(m):
                    if rngmark[i] == LP.UNCONST:
                        self.tier[n+i] = 1  # 4
                        n1+=1
                    elif rngmark[i] == LP.INFINITE:
                        self.tier[n+i] = 1  # 1
                    else:
                        self.tier[n+i] = 1  # 3
                        n1+=1

            else:
                for j in range(n):
                    if bndmark[j] != LP.FREEVAR:
                        self.tier[j] = 1  # 1
                    else:
                        self.tier[j] = 1  # 3
                        n1+=1

                for i in range(m):
                    if rngmark[i] == LP.UNCONST:
                        self.tier[n+i] = 1  # 4
                    elif rngmark[i] == LP.INFINITE:
                        self.tier[n+i] = 0  # 0
                    else:
                        self.tier[n+i] = 1  # 2


        #/*---------------------------------------------------------+
        #| compute maximum column degree of tier zero columns      */

        if self.dense < 0:
            denfac = 3.0
            colhisto = np.zeros(n+m+1, dtype=np.int)

            for i in range(n+m):
                if self.tier[i] == 0:
                    colhisto[ degree[i] ] += 1

            tot = 0
            _max = n1
            for i in range(n+m):
                tot += colhisto[i]
                if tot >= _max:
                    break
            i+=1
            tot = 0
            cnt = 0
            for j in range(n+m):
                if self.tier[j] == 0:
                    tot += degree[j]
                    cnt+=1
            dense = int(denfac*i)

            #dense = (int)(denfac*MAX(i,tot/cnt))
    		#printf("i = %d, n = %d, m = %d, n1 = %d \n", i,n,m,n1)
    		#printf("tot = %d, cnt = %d\n", tot, cnt)
            del(colhisto)


        if verbose>2:
            print("dense:                 {:5d}".format(dense))

        #/*----------------------------------------------+
        #| Get memory for mark[].                       */

        self.mark = np.empty(n+m, dtype=np.int)

        self.lltsym(degree,nbrs)

        del(degree)
        del(nbrs)
        self.tier = None

    def lltsym(self, degree, nbrs):
        dense = self.dense
        method = self.method
        stablty = self.stablty
        m = self.n + self.m
        tier = self.tier
        verbose = self.verbose

        tag = 0
        i2 = 0
        maxcolkey = 0
        for i in range(m):
            if tier[i] == 0:
                maxcolkey = max( degree[i], maxcolkey )


        if verbose>2:
            print("max tier zero degree:  {:5d}".format(maxcolkey))

        penalty = stablty*m
        if method == self._MLF:
            penalty *= m

        if verbose>2:
            print("ordering penalty:      {:5.0f}".format(penalty))

        #/*---------------------------------------------------------+
        #| allocate space for perm and iperm.                      */

        perm = self.perm = np.zeros(m, dtype=np.int)
        iperm = self.iperm = np.zeros(m, dtype=np.int)

        #/*---------------------------------------------------------+
        #| allocate space for work arrays.                         */

        dst = np.zeros(m, dtype=np.int)
        spc = np.zeros(m, dtype=np.int)
        locfil = np.zeros(m, dtype=np.int)
        hkey = np.zeros(m, dtype=np.int)
        heap = np.zeros(m+1, dtype=np.int)
        iheap = np.zeros(m, dtype=np.int)
        iwork = np.zeros(m, dtype=np.int)
        iwork2 = np.zeros(m, dtype=np.int)

        # heap--         /* so that indexing starts from 1 */
        # Above is implemented by having heap 1 larger.

        #/*---------------------------------------------------------+
        #| calculate the number of nonzeros in A.                  */

        aatnz = 0
        for i in range(m):
            aatnz += degree[i]
        aatnz = aatnz/2
        lnz   = aatnz

        #/*---------------------------------------------------------+
        #| allocate enough space to store symbolic structure of L   |
        #| (without any fillin).                                   */

        kAAt = self.kAAt = np.zeros(m+1, dtype=np.int)
        iAAt = self.iAAt = np.zeros(lnz, dtype=np.int)

        #/*----------------------------------------------+
        #| To reduce the number of REALLOC's, a separate |
        #| array spc[] is set up which tells how much    |
        #| space has been allocated for each node so far |
        #| (hence, spc[i] will always be >= degree[i]). */

        for i in range(m):
            spc[i] = degree[i]

        #/*---------------------------------------------------------+
        #| miscellaneous initializations.                          */

        for i in range(m):
            perm[i] = -1
            iperm[i] = -1
            iwork[i] = 0
            iwork2[i] = -1

        #/*---------------------------------------------------------+
        #| compute local fill for each node                        */

        if method == self._MLF:
            for node in range(m):
                lf = 0
                deg = degree[node]
                node_nbrs = nbrs[node]
                for k in range(deg):
                    nbr = node_nbrs[k]
                    nbr_nbrs = nbrs[nbr]
                    nbr_deg  = degree[nbr]
                    tag+=1
                    for kk in range(nbr_deg):
                        iwork[nbr_nbrs[kk]] = tag
                    for kk in range(k+1, deg):
                        if iwork[node_nbrs[kk]] != tag:
                            lf+=1

                locfil[node] = lf




        #/*---------------------------------------------------------+
        #| Hkey determines the basis for our ordering heuristic.    |
        #| For example, if hkey[node] = degree[node], then the code |
        #| will generate a minimum degree ordering.  Implicit in    |
        #| hkey is the tie-breaking rule - currently the rule       |
        #| is somewhat random.  To make it first occuring minimum,  |
        #| change the formula to:                                   |
        #|       hkey[node] = degree[node]*m + node                |
        #| warning: with this definition of hkey, there is the      |
        #| possibility of integer overflow on moderately large      |
        #| problems.                                                |
        #|                                                          |
        #| Nodes from the last m are assigned a penalty, as are     |
        #| nodes corresponding to dense columns.                    |
        #|                                                         */

        for node in range(m):
            if method == self._MD:
                hkey[node] = degree[node]
            elif method == self._MLF:
                hkey[node] = locfil[node]
            else:
                hkey[node] = node

        for node in range(m):
            if degree[node] > dense and tier[node] == 0:
                tier[node] = 1

            hkey[node] += tier[node]*penalty


        #/*---------------------------------------------------------+
        #| set up heap structure for quickly accessing minimum.    */

        heapnum = m
        for node in range(m-1, -1, -1):
            cur = node+1
            iheap[node] = cur
            heap[cur] = node
            hfall( heapnum, hkey, iheap, heap, cur )

        #/*---------------------------------------------------------+
        #| the min degree ordering loop                             |
        #|                                                         */

        tag = 0
        nz = 0
        i = 0
        kAAt[0] = 0

        self.denwin = m
        while i<m:
            # compute min hkey and find node achieving the min */
            node = heap[1]
            deg = degree[node]
            node_nbrs = nbrs[node]

            if ( deg >= m-1-i ):
                self.denwin = i

            # mark nodes for elimination: the min hkey node
            #   and any node 'indistinguishable' from it */

            perm[i] = node
            iperm[node] = i
            nd = 0
            i2 = i+1
            for k in range(deg):
                iperm[node_nbrs[k]] = i
            for k in range(deg):
                nbr = node_nbrs[k]
                if (degree[nbr] == deg and tier[nbr] == tier[node]):
                    nbr_nbrs = nbrs[nbr]
                    for kk in range(deg):
                        if (iperm[nbr_nbrs[kk]] < i):
                            break
                    kk += 1
                    if (kk == deg):
                        perm[i2] = nbr
                        iperm[nbr] = i2
                        i2+=1
                    else:
                        dst[nd] = nbr
                        nd+=1

                else:
                    dst[nd] = nbr
                    nd+=1

            # reallocate space for iAAt as necessary */

            ni = i2-i      # number of indistinguishable nodes */

            cnt = nz + ( deg*(deg+1) - (deg-ni)*(deg-ni+1) )/2
            if (cnt > lnz):
                lnz = max( cnt, 2*lnz )
                #iAAt = np.zeros(lnz, dtype=np.int)
                self.iAAt = iAAt = np.resize(iAAt,lnz)

            # copy adjacency lists in iAAt, kAAt */

            for ii in range(i, i2):
                node = perm[ii]
                node_nbrs = nbrs[node]
                kAAt[ii+1] = kAAt[ii] + deg

                for k in range(degree[node]):
                    nbr = node_nbrs[k]
                    row = iperm[nbr]
                    if ( row > ii ):
                        iAAt[nz] = nbr
                        nz+=1
                    elif ( row == i ):
                        if (nbr != perm[i]):
                            iAAt[nz] = nbr
                            nz+=1

                deg-=1

            # decrement degrees for each distinguishable
            #   neighbor of 'node' corresponding to the removal of
            #   'node' and the indistinguishable neighbors */

            node = perm[i]
            for k in range(nd):
                nbr = dst[k]
                nbr_nbrs = nbrs[nbr]
                degree[nbr]-=1
                nbr_deg = degree[nbr]
                kk = 0
                while nbr_nbrs[kk] != node:
                    kk+=1
                for kk in range(kk,nbr_deg):
                    nbr_nbrs[kk] = nbr_nbrs[kk+1]

            if (i2 > i+1):
                for k in range(nd):
                    nbr = dst[k]
                    nbr_nbrs = nbrs[nbr]
                    nbr_deg  = degree[nbr]
                    cnt = 0
                    for kk in range(nbr_deg):
                        if (iperm[nbr_nbrs[kk]] > i):
                            cnt+=1
                        else:
                            nbr_nbrs[kk-cnt] = nbr_nbrs[kk]

                    degree[nbr] -= cnt

            for ii in range(i, i2):
                node = perm[ii]

                cur = iheap[node]
                okey = hkey[heap[cur]]
                heap[cur] = heap[heapnum]
                iheap[heap[cur]] = cur
                heapnum-=1
                if (okey < hkey[heap[cur]]):
                    hfall(heapnum, hkey, iheap, heap, cur)
                else:
                    hrise(hkey, iheap, heap, cur)


            # add links: between each pair of distinguishable
            #              nodes adjacent to min-deg node which don't
            #              already have a link */

            if ( method != self._MLF or locfil[perm[i]] > 0 ):

                #if (nd > 1) cnt = memfree / (nd*(nd-1))

                for k in range(nd):
                    nbr = dst[k]
                    nbr_deg = degree[nbr]
                    nbr_nbrs = nbrs[nbr]
                    tag+=1
                    for kk in range(nbr_deg):
                        iwork[nbr_nbrs[kk]] = tag

                    for kk in range(k+1, nd):
                        nbr2 = dst[kk]
                        if (iwork[nbr2]!=tag):

                            if (method == self._MLF):
                                cnt2 = 0
                                ne = degree[nbr2]
                                for kkk in range(ne):
                                    nbr3 = nbrs[nbr2][kkk]
                                    if (iwork[nbr3] == tag):
                                        locfil[nbr3]-=1
                                        hkey[nbr3] = locfil[nbr3]
                                        if (tier[nbr3] != 0):
                                            hkey[nbr3] += tier[nbr3]*penalty
                                        hrise(hkey, iheap, heap, iheap[nbr3])
                                        cnt2+=1

                                locfil[nbr]  += degree[nbr]  - cnt2
                                locfil[nbr2] += degree[nbr2] - cnt2


                            ne = degree[nbr]
                            if (ne >= spc[nbr]):
                                #spc[nbr] = MAX(spc[nbr]+cnt+1,2*spc[nbr])
                                #memfree -= cnt

                                spc[nbr] *= 2
                                #nbrs[nbr] = np.zeros(spc[nbr], dtype=np.int)
                                nbrs[nbr].resize(spc[nbr])

                            nbrs[nbr][ne] = nbr2
                            degree[nbr]+=1
                            if (method == self._MLF):
                                iwork[nbr2] = tag

                            ne = degree[nbr2]
                            if (ne >= spc[nbr2]):
                                #spc[nbr2] = MAX(spc[nbr2]+cnt+1,2*spc[nbr2])
                                #memfree -= cnt

                                spc[nbr2] *= 2
                                #nbrs[nbr2] = np.zeros(spc[nbr2], dtype=np.int)
                                nbrs[nbr2].resize(spc[nbr2])

                            nbrs[nbr2][ne] = nbr
                            degree[nbr2]+=1

            # adjust heap */

            for k in range(nd):
                nbr = dst[k]
                if (method == self._MD):
                    hkey[nbr] = degree[nbr]
                elif (method == self._MLF):
                    locfil[nbr] -= ni*(degree[nbr]-nd+1)
                    hkey[nbr] = locfil[nbr]
                else:
                    hkey[nbr] = nbr

                if (tier[nbr] != 0):
                    hkey[nbr] += tier[nbr]*penalty
                hrise( hkey, iheap, heap, iheap[nbr] )
                hfall( heapnum, hkey, iheap, heap, iheap[nbr] )

            for ii in range(i, i2):
                node = perm[ii]
                nbrs[node] = None  # memfree += spc[node]
            i = i2

        if (verbose>2):
            print("size of dense window = {:d}".format(m - self.denwin))

        #heap+=1
        del(dst)
        del(spc); del(locfil); del(hkey); del(heap); del(iheap)
        del(iwork); del(iwork2)

        for k in range(kAAt[m]):
            iAAt[k] = iperm[iAAt[k]]

        for i in range(m):
            qksort(iAAt, kAAt[i], kAAt[i+1]-1)

        #/*---------------------------------------------------------+
        #| calculate and print statistics.                         */

        narth = 0.0e0
        for i in range(m):
            k = kAAt[i+1]-kAAt[i]
            narth += float(k*k)

        narth = narth + 3*kAAt[m] + m

        lnz    = kAAt[m]
        #iAAt = np.zeros(lnz, dtype=np.int)
        self.iAAt = iAAt = np.resize(iAAt,lnz)
        if (verbose>1):
            print("nonzeros:    L {:10d},  arith_ops {:18.0f}".format(lnz, narth))

        #/*---------------------------------------------------------+
        #| allocate remaining storage.                             */

        self.AAt = np.empty(lnz)
        self.diag = np.empty(m)



def qksort(v, left,right):
    """
    *** qksort: sort v[left]...v[right] into increasing order ***
        reference: The C Programming Language,
                   Kernighan and Ritchie
                   2nd. edition, page 87.
    """
    if (left >= right):
        return  #  do nothing if array contains fewer than two elements

    swap(v, left, (left + right)/2)  # move partition elem
    last = left  # to v[left]
    for i in range(left+1, right): # partition
        if (v[i] < v[left]):
            last += 1
            swap(v, last, i)
    swap(v, left, last) # restore partition elem
    qksort(v, left, last-1)
    qksort(v, last+1, right)

def swap(v, i, j):
    """*** swap: interchange v[i] and v[j] */"""
    temp = v[i]
    v[i] = v[j]
    v[j] = temp

def hfall(heapnum, key, iheap, heap, cur):

    child = 2*cur
    while (child <= heapnum):
        if (child < heapnum and key[heap[child+1]] < key[heap[child]]):
            child+=1
        if (key[heap[cur]] > key[heap[child]]):
            swap(heap, cur, child)
            swap(iheap, heap[cur], heap[child])
            cur = child
            child = 2*cur
        else:
            break

def hrise(key, iheap, heap, cur):
    parent = cur/2
    while (parent > 0):
        if (key[heap[parent]] > key[heap[cur]]):
            swap(heap, cur, parent)
            swap(iheap, heap[cur], heap[parent])
            cur = parent
            parent = cur/2
        else:
            break
