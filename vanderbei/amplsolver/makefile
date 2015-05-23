# /****************************************************************
# Copyright (C) AT&T 1992, 1993, 1994
# All Rights Reserved
#
# Permission to use, copy, modify, and distribute this software and
# its documentation for any purpose and without fee is hereby
# granted, provided that the above copyright notice appear in all
# copies and that both that the copyright notice and this
# permission notice and warranty disclaimer appear in supporting
# documentation, and that the name of AT&T or any of its entities
# not be used in advertising or publicity pertaining to
# distribution of the software without specific, written prior
# permission.
#
# AT&T DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
# INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
# IN NO EVENT SHALL AT&T OR ANY OF ITS ENTITIES BE LIABLE FOR ANY
# SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
# ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
# THIS SOFTWARE.
# ****************************************************************/
#
.SUFFIXES: .c .o
CC = cc
CFLAGS = -O
SHELL=/bin/sh

# Add -DKR_headers to CFLAGS if your C compiler does not
# understand ANSI C function headers, e.g.
#	CFLAGS = -O -DKR_headers
# If things don't run right, you may need to change -O to -g
# so you can poke around with a debugger.
# You may also need to add -D_SIZE_T to CFLAGS, or to
# comment out the definition of size_t in nlp.h .

# For Sun systems, try
# CFLAGS = -O -DKR_headers -D_SIZE_T

# For DEC Alpha, try
# CFLAGS = -g -ieee_with_no_inexact -DLong=int -DX64_bit_pointers -DDouble_Align

# To make edag_reset() available, add -DEDAG_RESET to CFLAGS.
# Invoking "edag_reset();" zeros edaginfo and permits a new edagread().

.c.o:
	$(CC) -c $(CFLAGS) $*.c

all: arith.h amplsolver.a qpcheck.o version.o

a = atof.c b_search.c basename.c bscanf.c conval.c derprop.c dtoa1.c\
 dynlink.c edagread.c fpecatch.c funcadd0.c g_fmt.c jacdim.c\
 mach.c mip_pri.c mypow.c objval.c obj_prec.c punknown.c readsol.c rops.c\
 sprintf.c sscanf.c writesol.c wrtsol_.c xectim.c

amplsolver.a: $a
	$(CC) -c $(CFLAGS) $?
	x=`echo $? | sed 's/\.c/.o/g'` && ar ruv amplsolver.a $$x && rm $$x
	ranlib amplsolver.a || true
# If your system lacks ranlib, add a dummy ranlib to your
# search path, e.g.
#	exec true
# or just comment out the ranlib invocation above.

dtoa1.o: dtoa.c arith.h
conval.o jacdim.o objval.o qpcheck.o wrtsol_.o: jacdim.h nlp.h nlp1.h
derprop.o edagread.o linedagread.o mip_pri.o rops.o: nlp.h nlp1.h
dynlink.o: funcadd.h
edagread.o linedagread.o: nlp.h nlp1.h
edagread.o: op_type.hd dvalue.hd r_opn.hd
funcadd.o: funcadd.h
mach.o: arith.h
readsol.o writesol.o: nlp.h nlp1.h
rops.o: r_op.hd
qpcheck.o: r_qp.hd

linedagread.o: linedagread.c
	$(CC) -c $(CFLAGS) -DJust_Linear linedagread.c

arith.h: arith.h0
	# Copy arith.h0 to arith.h, then edit arith.h to activate the
	# appropriate #define line, as explained in the comments at the top
	# For systems with IBM-mainframe arithmetic, see README.
	exit 1

# If compiling dtoa1.c reveals that your system lacks float.h, malloc.h
# or memory.h, you could try
#
#	  make float.h
#
#         make malloc.h
# and/or
#         make memory.h
#
# as appropriate.

float.h: float.h0
	ln float.h0 float.h

malloc.h:
	echo 'extern char *malloc();' >malloc.h

memory.h:
	echo 'extern char *memcpy();' >memory.h

# make xsum.out to check for transmission errors.
# This assumes you have the xsum program, whose source
# you can get by asking research!netlib to
#	send xsum.c from f2c/src

xsum.out: xsum0.out
	xsum \
	README \
	arith.h.ibm \
	arith.h0 \
	atof.c \
	b_search.c \
	basename.c \
	bscanf.c \
	conval.c \
	derprop.c \
	dtoa.c \
	dtoa1.c \
	dvalue.hd \
	dynlink.c \
	edagread.c \
	float.h0 \
	fpecatch.c \
	funcadd.c \
	funcadd.h \
	funcadd0.c \
	g_fmt.c \
	jacdim.c \
	jacdim.h \
	linedagread.c \
	mach.c \
	mip_pri.c \
	mypow.c \
	nlp.h \
	nlp1.h \
	obj_prec.c \
	objval.c \
	op_type.hd \
	punknown.c \
	qpcheck.c \
	r_op.hd \
	r_opn.hd \
	r_qp.hd \
	readsol.c \
	rnd_prod.s \
	rops.c \
	sprintf.c \
	sscanf.c \
	writesol.c \
	wrtsol_.c \
	xectim.c \
	>xsum1.out
	cmp xsum0.out xsum1.out && mv xsum1.out xsum.out || diff xsum[01].out
