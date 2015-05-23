/****************************************************************
Copyright (C) AT&T 1992, 1993, 1994
All Rights Reserved

Permission to use, copy, modify, and distribute this software and
its documentation for any purpose and without fee is hereby
granted, provided that the above copyright notice appear in all
copies and that both that the copyright notice and this
permission notice and warranty disclaimer appear in supporting
documentation, and that the name of AT&T or any of its entities
not be used in advertising or publicity pertaining to
distribution of the software without specific, written prior
permission.

AT&T DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL AT&T OR ANY OF ITS ENTITIES BE LIABLE FOR ANY
SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
****************************************************************/

#include "jacdim.h"
#include "errno.h"	/* for errno */

#ifdef KR_headers
 extern int x0_check();
#else
 extern int x0_check(real *);
#endif

 static int
#ifdef KR_headers
NNPROB_chk(N, NPROB) fint *N; fint *NPROB;
#else
NNPROB_chk(fint *N, fint *NPROB)
#endif
{
	int i = *NPROB;
	if (i < 0 || i >= n_obj || *N != o_vars) {
		fprintf(stderr,
		"objval: got N = %ld, NPROB = %ld; expected N = %d, 0 <= NPROB < %d\n",
			(long)*N, (long)*NPROB, o_vars, n_obj);
		exit(1);
		}
	return i;
	}

 real
#ifdef KR_headers
objval_(N, X, NPROB, nerror) fint *N, *NPROB, *nerror; real *X;
#else
objval_(fint *N, real *X, fint *NPROB, fint *nerror)
#endif
{
	cde *d;
	register expr *e1;
	real f0, f1;
	int i;
	ograd **gr0;
	register ograd *gr;
	Jmp_buf err_jmp0;
	register real *Adjoints;

	if (*nerror >= 0) {
		err_jmp = &err_jmp0;
		i = setjmp(err_jmp0.jb);
		if (*nerror = i)
			return 0.;
		}
	i = NNPROB_chk(N, NPROB);
	want_deriv = 1;
	errno = 0;	/* in case f77 set errno opening files */
	x0_check(X);
	if (comb && !(x0kind & 3))
		comeval(0,comb);
	if (ncom0 > combc)
		comeval(combc, ncom0);
	if (comc1 < ncom1)
		com1eval(comc1, ncom1);
	x0kind |= 2;
	if (comc1)
		com1eval(0,comc1);
	co_index = -(i + 1);
	d = obj_de + i;
	gr0 = Ograd + i;
	e1 = d->e;
	f0 = (*e1->op)(e1);
	f1 = 0.;
	Adjoints = adjoints;
	for(gr = *gr0; gr; gr = gr->next)
		f1 += gr->coef * X[gr->varno];
	err_jmp = 0;
	return f0 + f1;
	}

 void
#ifdef KR_headers
objgrd_(N, X, NPROB, G, nerror) fint *N, *NPROB, *nerror; real *X, *G;
#else
objgrd_(fint *N, real *X, fint *NPROB, real *G, fint *nerror)
#endif
{
	cde *d;
	register expr_v *V;
	register expr *e1;
	ograd **gr0;
	register ograd *gr;
	register real *Adjoints;
	Jmp_buf err_jmp0;
	int L, *z;
	register int i;

	if (*nerror >= 0) {
		err_jmp = &err_jmp0;
		i = setjmp(err_jmp0.jb);
		if (*nerror = i)
			return;
		}
	i = NNPROB_chk(N, NPROB);
	want_deriv = 1;
	if (x0_check(X) || !(x0kind & 2)) {
		objval_(N,X,NPROB,nerror);
		if (*nerror)
			return;
		}
	errno = 0;	/* in case f77 set errno opening files */
	if (f_b)
		funnelset(f_b);
	if (f_o)
		funnelset(f_o);
	Adjoints = adjoints;
	d = obj_de + i;
	gr0 = Ograd + i;
	V = var_e;
	e1 = d->e;
	for(gr = *gr0; gr; gr = gr->next)
		Adjoints[gr->varno] = gr->coef;
	if (L = d->zaplen) {
		memset((char *)adjoints_nv1, 0, L);
		derprop(d->d);
		}
	if (zerograds) {	/* sparse gradients */
		z = zerograds[i];
		while((i = *z++) >= 0)
			G[i] = 0;
		}
	for(gr = *gr0; gr; gr = gr->next) {
		i = gr->varno;
		G[i] = Adjoints[i];
		}
	err_jmp = 0;
	}
