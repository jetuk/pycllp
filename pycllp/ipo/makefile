########################################################################
# The C compiler must be an ansi C compiler.  A C++ compiler is even
# better.  The quad precision package, QuadPrec, only works with a C++
# compiler.
#
# Uncomment one of the following two lines:
CC = gcc
#CC = CC -DQuadPrec    # In this case, type: make quadipo
########################################################################

C = ../common
A = ../amplsolver
Q = ../Quad

.SUFFIXES: .c .o

OPT = -O

.c.o:
	$(CC) -c $(OPT) -I. -I$C -I$A -I$Q $*.c

COMMON = $C/solve.o $C/iolp.o $C/hash.o $C/cputime.o $C/strdup.o \
	 $C/hook.o $C/tree.o $C/heap.o $C/linalg.o

########################################################################
# There are two input formats to ipo:  MPS files and AMPL files.
# Given an MPS file (which can be a pain to make), running ipo
# is easy.  Here's an example:
#       ipo ../data/mps/afiro.mps
# Given an AMPL model (which is relatively easy to make), one can
# invoke ipo from within AMPL.  Here's an example:
#       ampl                                                
#       option solver ipo;
#       include ../data/ampl/gamethy2.mod;
# 
# Ipo can be compiled with the AMPL interface either enabled or
# disabled.  Enabling the AMPL interface has obvious advantages.  The
# advantage of disabling the interface is that it might be easier
# to compile the code since there are then a lot fewer files to deal
# with.  To compile with the AMPL interface enabled, you must first go
# to $A and run 'make' there.
# 
# Uncomment one of the following two lines:
AMPL = $C/noamplio.o
#AMPL = $C/amplio.o $A/qpcheck.o $A/version.o $A/amplsolver.a
########################################################################

########################################################################
LU = ldlt.o
########################################################################

########################################################################
# Three versions of interior point method are available.  
# Uncomment one of the following three lines:

#METHOD = intpt.o	# primal-dual path-following
METHOD = hsd.o		# homogeneous self-dual predictor-corrector 
#METHOD = hsdls.o	# homogeneous self-dual long-step
########################################################################

COMMON = $C/common.a

ipo: $(AMPL) $(METHOD) $(LU) $(COMMON)
	$(CC) -o ipo $(AMPL) $(METHOD) $(LU) $(COMMON) -lm \
	#/usr/local/lib/libdmalloc.a 
	#-lmalloc 

quadipo: $(AMPL) $(METHOD) $(LU) $(COMMON)
	$(CC) -o ipo $(AMPL) $(METHOD) $(LU) $(COMMON) -lm $Q/Quad.o

ldlt.o:  ldlt.c  $C/lp.h  $C/myalloc.h  $C/macros.h
ipo.o:     ipo.c     $C/lp.h  $C/myalloc.h  $C/macros.h 
ipoa.o:    ipoa.c    $C/lp.h  $C/myalloc.h  $C/macros.h $A/jacdim.h $A/r_opn.hd
intpt.o:   intpt.c            $C/myalloc.h  $C/macros.h $C/linalg.h


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
