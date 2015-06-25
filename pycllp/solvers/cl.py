

from __future__ import print_function
import numpy as np
import pyopencl as cl
import os
from ..linalg import smx, atnum
from ..ldlt import LDLTFAC
from . import BaseCSCSolver
import numpy as np

EPS = 1.0e-12
MAX_ITER = 200


class ClHSDSolver(BaseCSCSolver):
    name = 'clhsd'

    def __init__(self, ctx, queue):
        self.ctx = ctx
        self.queue = queue

    def init(self, A, b, c, f=0.0):
        BaseCSCSolver.init(self, A, b, c, f=f)

        self.init_cl()



    def init_cl(self, verbose=0):
        """Setup problem on the cl context
        """
        ctx = self.ctx
        n = int(self.n)
        m = int(self.m)
        nz = len(self.Ai)
        nlp = self.nlp
        kA = self.Ak.astype(np.int32)
        iA = self.Ai.astype(np.int32)
        A = np.ascontiguousarray(self.A.astype(np.float32)[0,:])
        b = self.b.reshape(np.prod(self.b.shape)).astype(np.float32)
        c= self.c.reshape(np.prod(self.c.shape)).astype(np.float32)

        self.local_size = 1
        self.global_size = nlp*self.local_size

        # Allocate local work memory for arrays
        fsize = np.float32().nbytes
        isize = np.int32().nbytes
        self.l_fwork = cl.LocalMemory(fsize*(9*n+10*m))
        self.l_iwork = cl.LocalMemory(isize*(3*n+3*m))

        # arrays for A^T
        At = np.zeros(nz,dtype=np.float32)
        iAt = np.zeros(nz, dtype=np.int32)
        kAt = np.zeros(m+1, dtype=np.int32)

        # Verify input.

        if m < 20 and n < 20 and verbose > 0:
            AA = np.zeros((20, 20))
            for j in range(n):
                for k in range(kA[j], kA[j+1]):
                    AA[iA[k]][j] = A[k]

            print("A <= b:")
            for i in range(m):
                for j in range(n):
                    print(" {:5.1f}".format(AA[i][j]), end="")
                print("<= {:5.1f}".format(b[i]))
            print("\nc:")

            for j in range(n):
                print(" {:5.1f}".format(c[j]), end="")

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

        # 	Display Banner.
        if verbose > 0:
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
        self.diag = np.zeros(self.diag.shape[0]*nlp, dtype=np.float32)
        self.g_diag = cl.Buffer(ctx, mf.READ_WRITE, self.diag.nbytes)

        self.perm = ldltfac.perm.astype(np.int32)
        self.g_perm = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=self.perm)

        self.iperm = ldltfac.iperm.astype(np.int32)
        self.g_iperm = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=self.iperm)

        self.AAt = ldltfac.AAt.astype(np.float32)
        self.lnz = self.AAt.shape[0]
        self.AAt = np.zeros(self.AAt.shape[0]*nlp, dtype=np.float32)
        self.g_AAt = cl.Buffer(ctx, mf.READ_WRITE, self.AAt.nbytes)
        #self.l_AAt = cl.LocalMemory(self.AAt.nbytes)

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
        if verbose > 0:
            print('Creating OpenCL program...')
        path = os.path.dirname(__file__)
        path = os.path.join(path, '..','cl')
        build_opts = ['-I '+path, ]#'-cl-single-precision-constant',
            #'-cl-opt-disable', ]

        src_files = ['hsd.cl', 'linalg.cl', 'ldlt.cl']
        src = ''
        for src_file in src_files:
            src += open(os.path.join(path,src_file)).read()

        self.cl_prg = cl.Program(ctx, src).build(options=build_opts)


    def solve(self,verbose=0):
        ctx = self.ctx
        queue = self.queue

        ls = (self.local_size,)
        gs = (self.global_size,)
        v = np.int32(verbose)

        m = np.int32(self.m)
        n = np.int32(self.n)
        lnz = np.int32(self.lnz)
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
        AAt = self.g_AAt
        iAAt = self.g_iAAt
        kAAt = self.g_kAAt
        #Q = self.g_Q
        #iQ = self.g_iQ
        #kQ = self.g_kQ

        diag = self.g_diag
        perm = self.g_perm
        iperm = self.g_iperm

        fwork = self.l_fwork
        iwork = self.l_iwork
        status = self.g_status
        if verbose > 0:
            print('Executing kernel...')
        self.cl_prg.hsd(
            queue,
            gs,
            ls,
            m,n,lnz,denwin,c,b,x,z,y,w,diag,perm,iperm,
            A,iA,kA,At,iAt,kAt,AAt,iAAt,kAAt,#Q,iQ,kQ,
            fwork,iwork,status,v
        )
        cl.enqueue_copy(queue, self.status, status)
        cl.enqueue_copy(queue, self.x, x)
        cl.enqueue_copy(queue, self.y, y)

        self.x = self.x.reshape((len(self.status),n))
        self.y = self.y.reshape((len(self.status),m))

        return (self.status,
            self.x, self.y)
