

from __future__ import print_function
import numpy as np
import pyopencl as cl
import os
from ..linalg import smx, atnum
from ..ldlt import LDLTFAC
from . import BaseCSCSolver
import time

EPS = 1.0e-12
MAX_ITER = 200


class ClHSDSolver(BaseCSCSolver):
    name = 'clhsd'

    def __init__(self, ctx, queue):
        self.ctx = ctx
        self.queue = queue
        self.total_buffer_size = 0
        self.total_buffers = 0

    def init(self, A, b, c, f=0.0):
        BaseCSCSolver.init(self, A, b, c, f=f)

        self.init_cl()

    def _make_buffer(self, *args):
        """A wrapper around cl.Buffer that increments total size"""
        buffer = cl.Buffer(*args)
        self.total_buffer_size += buffer.get_info(cl.mem_info.SIZE)
        self.total_buffers += 1
        return buffer

    def init_cl(self, verbose=1):
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

        # Host side storage of results.
        self.x = np.ones(nlp*n, dtype=np.float32)
        self.y = np.ones(nlp*m, dtype=np.float32)
        xsize = self.x.nbytes
        ysize = self.y.nbytes

        atnum(m,n,kA,iA,A,kAt,iAt,At)

        # List of tuple (source, destination) of data to copy to device
        # COPY_HOST_PTR is not used so we can benchmark the data transfer time
        data_to_transfer = []

        # Create buffers (but no data transfer)
        mf = cl.mem_flags

        self.g_c = self._make_buffer(ctx, mf.READ_ONLY, xsize)
        self.g_b = self._make_buffer(ctx, mf.READ_ONLY, ysize)
        data_to_transfer.extend([(c, self.g_c), (b, self.g_b)])

        # TODO copy initial conditions to device. CL code currently
        # starts at unity for all columns.
        self.g_x = self._make_buffer(ctx, mf.READ_WRITE, xsize)
        self.g_z = self._make_buffer(ctx, mf.READ_WRITE, xsize)
        self.g_w = self._make_buffer(ctx, mf.READ_WRITE, ysize)
        self.g_y = self._make_buffer(ctx, mf.READ_WRITE, ysize)

        self.g_iA = self._make_buffer(ctx, mf.READ_ONLY, iA.nbytes)
        self.g_kA = self._make_buffer(ctx, mf.READ_ONLY, kA.nbytes)
        self.g_A = self._make_buffer(ctx, mf.READ_ONLY, A.nbytes)
        data_to_transfer.extend([(iA, self.g_iA), (kA, self.g_kA), (A, self.g_A)])
        # buffers for A^T
        self.g_iAt = self._make_buffer(ctx, mf.READ_ONLY, iAt.nbytes)
        self.g_kAt = self._make_buffer(ctx, mf.READ_ONLY, kAt.nbytes)
        self.g_At = self._make_buffer(ctx, mf.READ_ONLY, At.nbytes)
        data_to_transfer.extend([(iAt, self.g_iAt), (kAt, self.g_kAt), (At, self.g_At)])

        self.status = np.empty(nlp, dtype=np.int32)
        self.g_status = self._make_buffer(ctx, mf.WRITE_ONLY, self.status.nbytes)

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
        self.g_diag = self._make_buffer(ctx, mf.READ_WRITE, self.diag.nbytes)

        self.perm = ldltfac.perm.astype(np.int32)
        self.g_perm = self._make_buffer(ctx, mf.READ_ONLY, self.perm.nbytes)
        data_to_transfer.append((self.perm, self.g_perm))

        self.iperm = ldltfac.iperm.astype(np.int32)
        self.g_iperm = self._make_buffer(ctx, mf.READ_ONLY, self.iperm.nbytes)
        data_to_transfer.append((self.iperm, self.g_iperm))

        self.AAt = ldltfac.AAt.astype(np.float32)
        self.lnz = self.AAt.shape[0]
        self.AAt = np.zeros(self.AAt.shape[0]*nlp, dtype=np.float32)
        self.g_AAt = self._make_buffer(ctx, mf.READ_WRITE, self.AAt.nbytes)
        #self.l_AAt = cl.LocalMemory(self.AAt.nbytes)

        self.iAAt = ldltfac.iAAt.astype(np.int32)
        self.g_iAAt = self._make_buffer(ctx, mf.READ_ONLY, self.iAAt.nbytes)
        data_to_transfer.append((self.iAAt, self.g_iAAt))

        self.kAAt = ldltfac.kAAt.astype(np.int32)
        self.g_kAAt = self._make_buffer(ctx, mf.READ_ONLY, self.kAAt.nbytes)
        data_to_transfer.append((self.kAAt, self.g_kAAt))

        # Q matrix is currently not used
        #self.Q = ldltfac.Q.astype(np.float32)
        #self.g_Q = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
        #                       hostbuf=self.Q)

        #self.iQ = ldltfac.iQ.astype(np.int32)
        #self.g_iQ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
        #                        hostbuf=self.iQ)

        #self.kQ = ldltfac.kQ.astype(np.int32)
        #self.g_kQ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
        #                        hostbuf=self.kQ)

        # Copy data from host to device
        ntrans, trans_bytes = 0, 0
        start = time.clock()
        with cl.CommandQueue(ctx) as queue:
            for src, dest in data_to_transfer:
                evt = cl.enqueue_copy(queue, dest, src, is_blocking=False)
                trans_bytes += src.nbytes
                ntrans += 1
        end = time.clock()

        if verbose > 0:
            lsize = self.l_fwork.size + self.l_iwork.size
            print("{} KiB of LocalMemory allocated for each work group ({} KiB in total).".format(
                  lsize/1024, lsize*self.global_size/1024))
            print("A total of {} buffers allocated with {} KiB.".format(
                  self.total_buffers, self.total_buffer_size/1024))
            print("Transfered {} KiB in {} seconds ({:.2f} KiB/s) using {} copies.".format(
                  trans_bytes/1024, end-start, trans_bytes/1024/(end-start), ntrans))


        if verbose > 0:
            print('Creating OpenCL program...')
        path = os.path.dirname(__file__)
        path = os.path.join(path, '..','cl')
        build_opts = ['-I '+path]#'-cl-single-precision-constant',
            #'-cl-opt-disable', ]

        src_files = ['hsd.cl', 'linalg.cl', 'ldlt.cl']
        src = ''
        for src_file in src_files:
            src += open(os.path.join(path,src_file)).read()

        self.cl_prg = cl.Program(ctx, src).build(options=build_opts)

        if verbose > 0:
            for device in ctx.get_info(cl.context_info.DEVICES):
                knl_wg_size = self.cl_prg.hsd.get_work_group_info(
                                cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
                knl_lm_size = self.cl_prg.hsd.get_work_group_info(
                                cl.kernel_work_group_info.LOCAL_MEM_SIZE, device)
                print("Maximum work group size of {} for hsd kernel for device {}".format(knl_wg_size, device.name))
                print("Local memory size of {} for hsd kernel for device {}".format(knl_lm_size, device.name))
                print("Maximum memory allocation size: {}".format(device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)))


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
        start = time.clock()
        with cl.CommandQueue(ctx) as queue:
            event = self.cl_prg.hsd(
                queue,
                gs,
                ls,
                m,n,lnz,denwin,c,b,x,z,y,w,diag,perm,iperm,
                A,iA,kA,At,iAt,kAt,AAt,iAAt,kAAt,#Q,iQ,kQ,
                fwork,iwork,status,v
            )
            event.wait()
            mid = time.clock()

            cl.enqueue_copy(queue, self.status, status)
            cl.enqueue_copy(queue, self.x, x)
            cl.enqueue_copy(queue, self.y, y)
        end = time.clock()

        if True:
            print("Executed kernel in {} seconds.".format(mid-start))
            print("Transferred results to host in {} seconds.".format(end-mid))

        self.x = self.x.reshape((len(self.status),n))
        self.y = self.y.reshape((len(self.status),m))

        return (self.status,
            self.x, self.y)
