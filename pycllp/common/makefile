########################################################################
# The C compiler must be an ansi C compiler.  A C++ compiler is even
# better.  The quad precision package, QuadPrec, only works with a C++
# compiler.
#
# Uncomment one of the following two lines:
CC = gcc
#CC = CC -DQuadPrec    # In this case, type: make quadsimpo
########################################################################

A = ../amplsolver
Q = ../Quad

.SUFFIXES: .c .o

OPT = -g
AR = ar
ARFLAGS = vq

.c.o:
	$(CC) -c $(OPT) -I. $*.c

COMMON = cputime.o hash.o heap.o hook.o iolp.o linalg.o main.o \
solve.o strdup.o tree.o 

all: common.a amplio.o noamplio.o

common.a: $(COMMON)
	-rm common.a
	$(AR) $(ARFLAGS) common.a $(COMMON)

amplio.o:  amplio.c  lp.h myalloc.h
	$(CC) -c $(OPT) -I. -I$Q -I$A amplio.c
noamplio.o:  noamplio.c  
	$(CC) -c $(OPT) -I. -I$Q -I$A noamplio.c

cputime.o:    cputime.c    
	$(CC) -c $(OPT) -I. cputime.c
hash.o:    hash.c    hash.h
	$(CC) -c $(OPT) -I. hash.c
heap.o:    heap.c    heap.h
	$(CC) -c $(OPT) -I. heap.c
hook.o:    hook.c
	$(CC) -c $(OPT) -I. hook.c
iolp.o:  iolp.c  myalloc.h
	$(CC) -c $(OPT) -I. -I$Q iolp.c
linalg.o:  linalg.c  linalg.h myalloc.h  macros.h 
	$(CC) -c $(OPT) -I. -I$Q linalg.c
main.o:  main.c  lp.h myalloc.h
	$(CC) -c $(OPT) -I. -I$Q main.c
solve.o:   solve.c   lp.h  myalloc.h  lp1.h
	$(CC) -c $(OPT) -I. -I$Q solve.c
strdup.o:    strdup.c    
	$(CC) -c $(OPT) -I. strdup.c
tree.o:    tree.c    tree.h
	$(CC) -c $(OPT) -I. tree.c
