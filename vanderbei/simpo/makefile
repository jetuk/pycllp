########################################################################
# The C compiler must be an ansi C compiler.  A C++ compiler is even
# better.  The quad precision package, QuadPrec, only works with a C++
# compiler.
#
# Uncomment one of the following two lines:
CC = CC
#CC = CC -DQuadPrec    # In this case, type: make quadsimpo
########################################################################

C = ../common
A = ../amplsolver
Q = ../Quad

.SUFFIXES: .c .o

OPT = -g

.c.o:
	$(CC) -c $(OPT) -I. -I$C -I$A -I$Q $*.c

COMMON = $C/solve.o $C/iolp.o $C/hash.o $C/cputime.o $C/strdup.o \
	 $C/hook.o $C/tree.o $C/heap.o $C/linalg.o

########################################################################
# There are two input formats to simpo:  MPS files and AMPL files.
# Given an MPS file (which can be a pain to make), running simpo
# is easy.  Here's an example:
#       simpo ../data/mps/afiro.mps
# Given an AMPL model (which is relatively easy to make), one can
# invoke simpo from within AMPL.  Here's an example:
#       ampl
#       option solver simpo;
#       include ../data/ampl/gamethy2.mod;
# 
# simpo can be compiled with the AMPL interface either enabled or
# disabled.  Enabling the AMPL interface has obvious advantages.  The
# advantage of disabling the interface is that it might be easier
# to compile the code since there are then a lot fewer files to deal
# with.  To compile with the AMPL interface enabled, you must first go
# to $A and run 'make' there.
#
# Uncomment one of the following two lines:
#AMPL = $C/noamplio.o
AMPL = $C/amplio.o $A/qpcheck.o $A/version.o $A/amplsolver.a
########################################################################

########################################################################
# There are two versions of the matrix factorization code.  One, found in
# lueta.c, uses eta matrices to solve systems of equations involving a 
# subsequent basis matrix.  The other, found in lurefac.c, refactors the
# basis matrix using the previous factorization as a starting point.
#
# Uncomment one of the following two lines:
LU = lueta.o
#LU = lurefac.o
########################################################################

########################################################################
# Two versions of the simplex method are available.  The two-phase, dual
# phase I followed by primal phase II, is found in 2phase.c.  The 
# primal-dual simplex method is found in pd.c.
#
# Uncomment one of the following two lines:
#METHOD = 2phase.o
METHOD = pd.o
########################################################################

COMMON = $C/common.a

simpo: $(AMPL) $(METHOD) $(LU) $(COMMON)
	$(CC) -o simpo $(AMPL) $(METHOD) $(LU) $(COMMON) -lm \
	#/usr/local/lib/libdmalloc.a 
	#-lmalloc 

quadsimpo: $(AMPL) $(METHOD) $(LU) $(COMMON)
	$(CC) -o simpo $(AMPL) $(METHOD) $(LU) $(COMMON) -lm $Q/Quad.o

lueta.o:   lueta.c   lu.h     $C/myalloc.h  $C/macros.h $C/linalg.h $C/heap.h $C/tree.h 
lurefac.o: lurefac.c lu.h     $C/myalloc.h  $C/macros.h $C/linalg.h $C/heap.h $C/tree.h
2phase.o:  2phase.c  lu.h     $C/myalloc.h  $C/macros.h $C/linalg.h
pd.o:      pd.c      lu.h     $C/myalloc.h  $C/macros.h $C/linalg.h
cc.o:      cc.c      lu.h     $C/myalloc.h  $C/macros.h $C/linalg.h
pp.o:      pp.c      lu.h     $C/myalloc.h  $C/macros.h $C/linalg.h

$A/amplsolver.a:
	( cd $A && $(MAKE) amplsolver.a )

$A/qpcheck.o:
	( cd $A && $(MAKE) qpcheck.o )

$A/version.o:
	( cd $A && $(MAKE) version.o )

$C/common.a:
	( cd $C && $(MAKE) common.a )

$C/amplio.o:
	( cd $C && $(MAKE) amplio.o )

$C/noamplio.o:
	( cd $C && $(MAKE) noamplio.o )
