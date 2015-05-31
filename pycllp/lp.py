"""
Original implementation:
    Copyright (c) Robert J. Vanderbei, 1994
    All Rights Reserved
    http://www.princeton.edu/~rvdb/LPbook/

Port to Python
    James E. Tomlinson
    2015
"""
import sys

class LP(object):

    HEADER =0
    NAME   =1
    ROWS   =2
    COLS   =3
    RHS    =4
    RNGS   =5
    BNDS   =6
    QUADS  =7
    END    =8

    UNSET  =0
    PRIMAL =1
    DUAL   =2

    FINITE   =0x1
    INFINITE =0x2
    UNCONST  =0x4
    FREEVAR   =0x1
    BDD_BELOW =0x2
    BDD_ABOVE =0x4
    BOUNDED   =0x8

    #LP_OPEN_MAX 20  # max #	lp problems open at once

    def __init__(self, m, n, nz, iA, kA, A, b, c, ):

        self.m = m
        self.n = n
        self.nz = nz
        self.iA = iA
        self.kA = kA
        self.A = A
        self.b = b
        self.c = c

        
        #self.qnz = 0
        #self.name   =''
        #self.obj    =''
        #self.rhs    =''
        #self.bounds =''
        #self.ranges =''
        #self.f = 0.0
        #self.r =	None
        #self.l =	None
        #self.u =	None
        #self.iQ = None
        #self.kQ = None
        #self.Q =	None
        #self.w = None
        #self.x = None
        #self.y = None
        #self.z = None
        #self.kAt = None
        #self.iAt = None
        #self.At  = None
        #self.rowlab = None
        #self.collab = None
        #self.varsgn = None
        #self.tier = None
        #self.max	    = 1     #	max = -1,   min	= 1
        #self.sf_req  = 8     #	significant figures requested
        #self.itnlim  = 200   # iteration limit
        #self.timlim  = sys.float_info.max # time limit
        #self.verbose = 2     #	verbosity level
        #self.inftol  = 1.0e-5#	infeasibility requested
        #self.init_vars	  = deflt_hook
        #self.h_init        = deflt_hook
        #self.h_update 	  = deflt_hook
        #self.h_step   	  = deflt_hook
