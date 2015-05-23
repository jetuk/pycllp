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

#ifdef __cplusplus
extern "C" {
#endif

#include "jacdim.h"

#ifdef KR_headers
#define Void /*void*/
#else
#define Void void
#endif
 extern void what_prog(Void);

 static size_t zap_J;

 int
#ifdef KR_headers
x0_check(X) register real *X;
#else
x0_check(register real *X)
#endif
{
	register expr_v *V;
	register int i;

	if (x0kind == 4 || memcmp((char *)Lastx, (char *)X, x0len)) {
		memcpy((char *)Lastx, (char *)X, x0len);
		V = var_e;
		for(i = c_vars; i-- > 0; V++)
			V->v = *X++;
		x0kind = 0;
		return 1;
		}
	return 0;
	}

 void
#ifdef KR_headers
conval_(M, N, X, F, nerror) fint *M, *N, *nerror; register real *X; real *F;
#else
conval_(fint *M, fint *N, register real *X, real *F, fint *nerror)
#endif
{
	cde *d, *dend;
	register expr *e1;
	real f0, f1;
	register int i;
	cgrad **gr0;
	register cgrad *gr;
	Jmp_buf err_jmp0;

	if (*M != n_con || *N != c_vars) {
		what_prog();
		fprintf(stderr,
		"conval: got M = %ld, N = %ld; expected M = %d, N = %d\n",
			(long)*M, (long)*N, n_con, c_vars);
		exit(1);
		}
	if (*nerror >= 0) {
		err_jmp = &err_jmp0;
		i = setjmp(err_jmp0.jb);
		if (*nerror = i)
			return;
		}
	want_deriv = 1;
	errno = 0;	/* in case f77 set errno opening files */
	x0_check(X);
	i = (x0kind & 2) ? comb : 0;
	if (i < combc) {
		comeval(i, combc);
		}
	if (comc1)
		com1eval(0,comc1);
	x0kind |= 1;
	d = con_de;
	dend = d + n_con;
	co_index = 0;
	for(gr0 = Cgrad; d < dend; d++, gr0++, co_index++) {
		e1 = d->e;
		f0 = (*e1->op)(e1);
		f1 = 0.;
		for(gr = *gr0; gr; gr = gr->next) {
			i = gr->varno;
			f1 += X[i] * gr->coef;
			}
		if (F)
			*F++ = f0 + f1;
		}
	err_jmp = 0;
	}

 static void
#ifdef KR_headers
mnnzchk(M, N, NZ, who1) fint *M, *N, *NZ; char *who1;
#else
mnnzchk(fint *M, fint *N, fint *NZ, char *who1)
#endif
{
	if (*M != n_con || *N != c_vars || *NZ != nzjac) {
		what_prog();
		fprintf(stderr,
 "%s: got M = %ld, N = %ld, NZ = %ld\nexpected M = %d, N = %d, NZ = %d\n",
			who1, (long)*M, (long)*N, *NZ, n_con, c_vars, nzjac);
		exit(1);
		}
	}

 void
#ifdef KR_headers
jacval_(M, N, NZ, X, G, nerror) fint *M, *N, *NZ, *nerror; real *X, *G;
#else
jacval_(fint *M, fint *N, fint *NZ, real *X, real *G, fint *nerror)
#endif
{
	cde *d, *dend;
	register expr_v *V;
	register expr *e1;
	cgrad **gr0;
	register cgrad *gr;
	register real *Adjoints;
	Jmp_buf err_jmp0;
	int L;

	mnnzchk(M, N, NZ, "jacval");
	if (*nerror >= 0) {
		err_jmp = &err_jmp0;
		L = setjmp(err_jmp0.jb);
		if (*nerror = L)
			return;
		}
	want_deriv = 1;
	if (x0_check(X) || !(x0kind & 1))
		conval_(M,N,X,0,nerror);
	errno = 0;	/* in case f77 set errno opening files */
	if (zap_J)
		memset((char *)G, 0, zap_J);
	Adjoints = adjoints;
	d = con_de;
	dend = d + n_con;
	V = var_e;
	if (f_b)
		funnelset(f_b);
	if (f_c)
		funnelset(f_c);
	for(gr0 = Cgrad; d < dend; d++, gr0++) {
		for(gr = *gr0; gr; gr = gr->next)
			Adjoints[gr->varno] = gr->coef;
		e1 = d->e;
		if (L = d->zaplen) {
			memset((char *)adjoints_nv1, 0, L);
			derprop(d->d);
			}
		for(gr = *gr0; gr; gr = gr->next)
			G[gr->goff] = Adjoints[gr->varno];
		}
	err_jmp = 0;
	}

 static void
#ifdef KR_headers
LUcopy(nv, L, U, LU) int nv; register real *L, *U, *LU;
#else
LUcopy(int nv, register real *L, register real *U, register real *LU)
#endif
{
	register real *LUe;
	for(LUe = LU + 2*nv; LU < LUe; LU += 2) {
		*L++ = LU[0];
		*U++ = LU[1];
		}
	}

 void
#ifdef KR_headers
jacinc(JP, JI, X, LUp, LUrhsp, Inf)
	register fint *JP;
	register short *JI; real **X, **LUp, **LUrhsp, *Inf;
#else
jacinc(register fint *JP, register short *JI, real **X, real **LUp,
	real **LUrhsp, real *Inf)
#endif
{
	register fint n;
	cgrad **gr0;
	register cgrad *gr;

	*Inf = Infinity;
	gr0 = Cgrad + n_con;
	for(n = n_con; n > 0; --n)
		for(gr = *--gr0; gr; gr = gr->next) {
			JI[gr->goff] = (short)n;
			JP[gr->varno] = gr->goff + 1;
			}
	JP[c_vars] = nzjac + 1;
	*LUp = LUv;
	*LUrhsp = LUrhs;
	*X = X0;
	}

 void
#ifdef KR_headers
jacinc_(M, N, NZ, JP, JI, X, L, U, Lrhs, Urhs, Inf)
	fint *M, *N, *NZ; register fint *JP;
	register short *JI; real *X, *L, *U, *Lrhs, *Urhs, *Inf;
#else
jacinc_(fint *M, fint *N, fint *NZ, register fint *JP,
	register short *JI, real *X, real *L, real *U,
	real *Lrhs, real *Urhs, real *Inf)
#endif
{
	real *LU1, *LUrhs1, *x0;

	mnnzchk(M, N, NZ, "jacinc");
	jacinc(JP, JI, &x0, &LU1, &LUrhs1, Inf);
	if (n_con)
		LUcopy(n_con, Lrhs, Urhs, LUrhs1);
	LUcopy(c_vars, L, U, LU1);
	memcpy((char *)X, (char *)x0, x0len);
	}

 void
densej_(Void)	/* arrange for dense Jacobian computation */
{
	cgrad *cg;
	int i;

	if (nzc < n_con*n_var) {
		zap_J = n_con*n_var*sizeof(real);
		for(i = 0; i < n_con; i++)
			for(cg = Cgrad[i]; cg; cg = cg->next)
				cg->goff = i + n_con*cg->varno;
		}
	}

#ifdef __cplusplus
	}
#endif
