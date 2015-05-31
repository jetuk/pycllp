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
import pyopencl as cl
#from linalg import smx
from ..ldlt import LDLTFAC
import sys
import os

HUGE_VAL = sys.float_info.max


class LDLTFAC_CL(LDLTFAC):



    def init_cl(self, ctx, ):
        """
        Initialize CL parts of solution.
        """
        # System must be inverted first, same test as used in inv_num
        if self.diag is None:
            self.inv_sym()

        mf = cl.mem_flags
        # AAt
        g_AAt = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.AAt)
        g_iAAt = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.iAAt)
        g_kAAt = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.kAAt)


        print('Creating OpenCL program...')
        path = os.path.dirname(__file__)
        build_opts = '-I '+path

        src_files = ['linalg.cl', 'ldlt.cl']
        src = ''
        for src_file in src_files:
            src += open(os.path.join(path,src_file)).read()

        self.cl_prg = cl.Program(ctx, src).build(options=build_opts)


    def inv_num(self,
                dn,  # diagonal matrix for upper-left  corner
                dm  # diagonal matrix for lower-right corner
                ):
        pass
