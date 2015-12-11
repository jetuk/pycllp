
from . import BaseSolver
from pycllp.ldl import cl_krnl_ldl
from pycllp.cl_tools import cl_program_from_files, CL_PATH
import numpy as np
import pyopencl as cl
import time

DTYPE = np.float32
IDTYPE = np.int32

class ClDensePrimalNormalSolver(BaseSolver):
    """

    """
    name = 'cl_dense_primal_normal'

    def __init__(self, ctx=None, queue=None):
        super(ClDensePrimalNormalSolver, self).__init__()
        if ctx is None:
            ctx = cl.create_some_context()
        if queue is None:
            queue = cl.CommandQueue(ctx)
        self.ctx = ctx
        self.queue = queue
        self.buffers = {}

    def init(self, lp, verbose=0):
        if verbose > 0:
            print("Initializing ClDensePrimalNormalSolver solver...")
        ctx = self.ctx
        queue = self.queue
        mf = cl.mem_flags

        n = np.int32(lp.ncols)
        m = np.int32(lp.nrows)
        cl_size = lp.nproblems

        A = np.array(lp.A.todense()).astype(DTYPE)
        self._x = np.empty(n*cl_size, dtype=np.float64)
        self.status = np.empty(cl_size, dtype=IDTYPE)
        # Copy A to the device.
        # This should now never change!
        if verbose > 0:
            print("Creating buffers...")
        self.buffers['A'] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)

        # Create coordinate buffers
        for coord, size in zip(('x', 'z', 'y', 'w'), (n, n, m, m)):
            for pfx in ('', 'd'):
                self.buffers[pfx+coord] = cl.Buffer(ctx, mf.READ_WRITE, size*cl_size*np.float64().itemsize)

        # The constraint bounds and objective function buffers will only
        # ever be updated by the host to the device, and do not need to read
        self.buffers['b'] = cl.Buffer(ctx, mf.READ_WRITE | mf.HOST_WRITE_ONLY, m*cl_size*DTYPE().itemsize)
        self.buffers['c'] = cl.Buffer(ctx, mf.READ_WRITE | mf.HOST_WRITE_ONLY, n*cl_size*DTYPE().itemsize)

        size = m*(m+1)/2
        self.buffers['L'] = cl.Buffer(ctx, mf.READ_WRITE | mf.HOST_NO_ACCESS, size*cl_size*DTYPE().itemsize)
        self.buffers['D'] = cl.Buffer(ctx, mf.READ_WRITE | mf.HOST_NO_ACCESS, m*cl_size*DTYPE().itemsize)
        self.buffers['S'] = cl.Buffer(ctx, mf.READ_WRITE | mf.HOST_NO_ACCESS, m*cl_size*np.float64().itemsize)
        self.buffers['status'] = cl.Buffer(ctx, mf.WRITE_ONLY, cl_size*IDTYPE().itemsize)

        # Argument list for main solve routine, standard_primal_normal
        self.solve_args = [m, n]
        self.solve_args += [self.buffers[key] for key in ('A', 'x', 'z', 'y', 'w', 'dx', 'dz',
                            'dy', 'dw', 'b', 'c', 'L', 'D', 'S', 'status')]

        if verbose > 0:
            print("Building OpenCL program...")
        build_opts = ['-I '+CL_PATH, '-Werror',]
        self.program = cl_program_from_files(ctx, ('primal_normal.cl', 'ldl.cl')).build(options=build_opts)

        # Set the coordinate vectors to 1.0
        # This is only done at the beginning, solves after the first begin
        # where the previous solve finished.
        if verbose > 0:
            print("Initializing central path variables on device...")
        event = self.program.initialize_xzyw(queue, (cl_size,), None, m, n, *[self.buffers[key] for key in ('x', 'z', 'y', 'w')])
        event.wait()

        if verbose > 0:
            print("Solver initialized.")

    def solve(self, lp, verbose=0):
        ctx = self.ctx
        queue = self.queue
        if verbose > 0:
            print("Solving LP using ClDensePrimalNormalSolver...")

        n = np.int32(lp.ncols)
        m = np.int32(lp.nrows)
        cl_size = lp.nproblems


        # Copy new bounds and objective function to device
        if verbose > 0:
            print("Copying to 'b' vectors to device...")
        cl.enqueue_copy(queue, self.buffers['b'], np.ascontiguousarray(lp.b.T).astype(DTYPE))
        if verbose > 0:
            print("Copying to 'c' vectors to device...")
        cl.enqueue_copy(queue, self.buffers['c'], np.ascontiguousarray(lp.c.T).astype(DTYPE))
        # Solve the program
        if verbose > 0:
            print("Executing solver kernel...")
            t = time.time()

        event = self.program.initialize_xzyw(queue, (cl_size,), None, m, n, *[self.buffers[key] for key in ('x', 'z', 'y', 'w')])
        event.wait()

        event = self.program.standard_primal_normal(queue, (cl_size, ), None, *(self.solve_args+[np.int32(verbose)]))
        event.wait()
        if verbose > 0:
            print("Kernel complete in {} seconds.".format(time.time()-t))
            print("Copying 'x' vectors to host...")

        cl.enqueue_copy(queue, self._x, self.buffers['x'])
        self.x = self._x.reshape((n, cl_size)).T
        if verbose > 0:
            print("Copying 'status' vector to hose...")
        cl.enqueue_copy(queue, self.status, self.buffers['status'])

        if verbose > 0:
            print("Solve complete.")
