HEADER = 0
NAME = 1
ROWS =  2
COLS =  3
RHS  =  4
RNGS =  5
BNDS =  6
QUADS=  7
END  =  8

UNSET = 0
PRIMAL= 1
DUAL  = 2

FINITE    =0x1
INFINITE  =0x2
UNCONST   =0x4
FREEVAR   =0x1
BDD_BELOW =0x2
BDD_ABOVE =0x4
BOUNDED   =0x8

from . import lp
from . import solvers
