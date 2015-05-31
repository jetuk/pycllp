#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import pyopencl as cl
import os
from ..lp import LP
from ..linalg import smx, atnum
from ..ldlt import LDLTFAC

def small_problem():
    import numpy as np
    c = np.array([
        [2,3,4],
        [2,3,4],
        [2,3,4],
        [2,3,4]], dtype=np.float32)     # coefficients of variables in objective function.

    c *= 0.5+np.random.rand(*c.shape).astype(np.float32)

    A = np.array(
                [[3,2,1],   # Coefficient matrix,
                [2,5,3]], dtype=np.float32)   # Only for "<" constraint equations!

    b = np.array([
        [10, 15],
        [10, 15],
        [10, 15],
        [10, 15]], dtype=np.float32)       # Right hand side vector.
    m, n = A.shape
    A = A.T.flatten()
    nz = len(A)
    iA = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    kA = np.array([0, 2, 4, 6], dtype=np.int32)
    f = 0.001

    return m, n, nz, iA, kA, A, b, c, f


EPS = 1.0e-12
MAX_ITER = 200

class HSDLP_CL(LP):
    def __init__(self, m, n, nz, iA, kA, A, b, c, ):
        LP.__init__(self, m, n, nz, iA, kA, A, b, c,)
        assert b.ndim == 2
        assert c.ndim == 2
        assert b.shape[0] == c.shape[0]
        # Number of simultaneous problems each to be solved in a
        # cl workgroup
        self.nlp = b.shape[0]


    def init_cl(self, ctx, verbose=0):
        """Setup problem on the cl context
        """
        n = self.n
        m = self.m
        nz = self.nz
        nlp = self.nlp
        kA = self.kA
        iA = self.iA
        A = self.A
        b = self.b
        c= self.c

        self.local_size = 1
        self.global_size = nlp*self.local_size

        # Allocate local work memory for arrays
        fsize = np.float32().nbytes
        isize = np.int32().nbytes
        self.l_fwork = cl.LocalMemory(fsize*(12*n+12*m))
        self.l_iwork = cl.LocalMemory(isize*(4*n+4*m))

        # arrays for A^T
        At = np.zeros(nz,dtype=np.float32)
        iAt = np.zeros(nz, dtype=np.int32)
        kAt = np.zeros(n+1, dtype=np.int32)

        # Verify input.

        if m < 20 and n < 20:
            AA = np.zeros((20, 20))
            for j in range(n):
                for k in range(kA[j], kA[j+1]):
                    AA[iA[k]][j] = A[k]

            print("A <= b:")
            for i in range(m):
                for j in range(n):
                    print(" {:5.1f}".format(AA[i][j]), end="")
                print("<= {:5.1f}".format(b[0, i]))
            print("\nc:")

            for j in range(n):
                print(" {:5.1f}".format(c[0, j]), end="")

            print("")

        #  Initialization.

        self.x = np.ones(nlp*n, dtype=np.float32)
        z = np.ones(nlp*n, dtype=np.float32)
        w = np.ones(nlp*m, dtype=np.float32)
        self.y = np.ones(nlp*m, dtype=np.float32)

        atnum(m,n,kA,iA,A,kAt,iAt,At)

        # Initialize buffers
        mf = cl.mem_flags

        self.g_c = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
        self.g_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

        self.g_x = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.x)
        self.g_z = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=z)
        self.g_w = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=w)
        self.g_y = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.y)

        self.g_iA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=iA)
        self.g_kA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kA)
        self.g_A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        # buffers for A^T
        self.g_iAt = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=iAt)
        self.g_kAt = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kAt)
        self.g_At = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=At)

        self.status = np.empty(nlp, dtype=np.int32)
        self.g_status = cl.Buffer(ctx, mf.WRITE_ONLY, self.status.nbytes)

        print (kA, kAt)
        # 	Display Banner.

        print("m = {:d},n = {:d},nz = {:d}".format(m, n, nz))
        print(
    """--------------------------------------------------------------------------
             |           Primal          |            Dual           |       |
      Iter   |  Obj Value       Infeas   |  Obj Value       Infeas   |  mu   |
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    """)

        # 	Iteration.
        ldltfac = LDLTFAC(n, m, kAt, iAt, At, kA, iA, A, verbose)
        ldltfac.inv_sym()

        self.denwin = ldltfac.denwin
        # Create buffers from LDLT factorization.
        # Must convert to correct dtypes first

        self.diag = ldltfac.diag.astype(np.float32)
        self.g_diag = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=self.diag)
        print('diag',self.diag)
        self.perm = ldltfac.perm.astype(np.int32)
        self.g_perm = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=self.perm)

        self.iperm = ldltfac.iperm.astype(np.int32)
        self.g_iperm = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=self.iperm)
        print('iperm', self.iperm)
        self.AAt = ldltfac.AAt.astype(np.float32)
        #self.g_AAt = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
        #                       hostbuf=self.AAt)
        self.l_AAt = cl.LocalMemory(self.AAt.nbytes)

        self.iAAt = ldltfac.iAAt.astype(np.int32)
        self.g_iAAt = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=self.iAAt)

        self.kAAt = ldltfac.kAAt.astype(np.int32)
        self.g_kAAt = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=self.kAAt)

        self.Q = ldltfac.Q.astype(np.float32)
        self.g_Q = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=self.Q)

        self.iQ = ldltfac.iQ.astype(np.int32)
        self.g_iQ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=self.iQ)

        self.kQ = ldltfac.kQ.astype(np.int32)
        self.g_kQ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=self.kQ)


        print('Creating OpenCL program...')
        path = os.path.dirname(__file__)
        build_opts = '-I '+path

        src_files = ['hsd.cl', 'linalg.cl', 'ldlt.cl']
        src = ''
        for src_file in src_files:
            src += open(os.path.join(path,src_file)).read()

        self.cl_prg = cl.Program(ctx, src).build(options=build_opts)


    def solve(self, ctx, queue, f):

        ls = (self.local_size,)
        gs = (self.global_size,)

        m = np.int32(self.m)
        n = np.int32(self.n)
        denwin = np.int32(self.denwin)
        c = self.g_c
        b = self.g_b
        x = self.g_x
        z = self.g_z
        y = self.g_y
        w = self.g_w

        A = self.g_A
        iA = self.g_iA
        kA = self.g_kA
        At = self.g_At
        iAt = self.g_iAt
        kAt = self.g_kAt
        AAt = self.l_AAt
        iAAt = self.g_iAAt
        kAAt = self.g_kAAt
        Q = self.g_Q
        iQ = self.g_iQ
        kQ = self.g_kQ

        diag = self.g_diag
        perm = self.g_perm
        iperm = self.g_iperm

        fwork = self.l_fwork
        iwork = self.l_iwork
        status = self.g_status

        self.cl_prg.hsd(
            queue,
            gs,
            ls,
            m,n,denwin,c,b,x,z,y,w,diag,perm,iperm,
            A,iA,kA,At,iAt,kAt,AAt,iAAt,kAAt,Q,iQ,kQ,
            fwork,iwork,status
        )
        cl.enqueue_copy(queue, self.status, status)
        cl.enqueue_copy(queue, self.x, x)
        cl.enqueue_copy(queue, self.y, y)


        return (self.status,
            self.x.reshape((len(self.status),n)),
            self.y.reshape((len(self.status),m)))


if __name__ is '__main__':




    print('Creating CL context and queue...')
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    print(cl.device_info.MAX_WORK_ITEM_DIMENSIONS)
    print(cl.device_info.MAX_WORK_ITEM_SIZES)
    print(cl.device_info.MAX_WORK_GROUP_SIZE)

    m, n, nz, iA, kA, A, b, c, f = small_problem()

    print('Creating problem...')


    from pyipo.hsd import HSDLP
    args = small_problem()
    lp = HSDLP_CL(*args[:-1] )
    lp.init_cl(ctx,)
    status, x, y = lp.solve(ctx, queue, args[-1])

    for i, stat in enumerate(status):
        print('Solution for problem {}'.format(i))
        print('Status {:d}'.format(stat))
        print('x',x[i,:])
