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

#include "nlp.h"

#ifdef KR_headers
extern real mypow();
#else
extern real mypow(real,real);
#endif

 static void
#ifdef KR_headers
jmp_check(J) Jmp_buf *J;
#else
jmp_check(Jmp_buf *J)
#endif
{
	if (J)
		longjmp(J->jb, 1);
	}

 static void
#ifdef KR_headers
introuble(who,a) char *who; real a;
#else
introuble(char *who, real a)
#endif
{
	jmp_check(err_jmp);
	report_where();
	fprintf(stderr, "can't evaluate %s(%g)\n", who, a);
	jmp_check(err_jmp1);
	exit(1);
	}

 static void
#ifdef KR_headers
introuble2(who, a, b) char *who; real a, b;
#else
introuble2(char *who, real a, real b)
#endif
{
	jmp_check(err_jmp);
	report_where();
	fprintf(stderr, "can't evaluate %s(%g,%g)\n", who, a, b);
	jmp_check(err_jmp1);
	exit(1);
	}

 static real
#ifdef KR_headers
f_OPPLUS(e) register expr *e;
#else
f_OPPLUS(register expr *e)
#endif
{
	register expr *L;
	/* e->dL = e->dR = 1.; */
	L = e->L.e;
	e = e->R.e;
	return (*L->op)(L) + (*e->op)(e);
	}

 static real
#ifdef KR_headers
f_OPMINUS(e) register expr *e;
#else
f_OPMINUS(register expr *e)
#endif
{
	register expr *L;
	/* e->dL = 1.;  */
	/* e->dR = -1.; */
	L = e->L.e;
	e = e->R.e;
	return (*L->op)(L) - (*e->op)(e);
	}

 static real
#ifdef KR_headers
f_OPMULT(e) register expr *e;
#else
f_OPMULT(register expr *e)
#endif
{
	register expr *e1, *e2;
	e1 = e->L.e;
	e2 = e->R.e;
	return (e->dR = (*e1->op)(e1)) * (e->dL = (*e2->op)(e2));
	}

 static void
#ifdef KR_headers
zero_div(L, op) real L; char *op;
#else
zero_div(real L, char *op)
#endif
{
	errno = EDOM;
	jmp_check(err_jmp);
	report_where();
	fprintf(stderr, "can't compute %g%s0\n", L, op);
	jmp_check(err_jmp1);
	exit(1);
	}

 static real
#ifdef KR_headers
f_OPDIV(e) register expr *e;
#else
f_OPDIV(register expr *e)
#endif
{
	register expr *e1;
	real L, R, rv;
	e1 = e->L.e;
	L = (*e1->op)(e1);
	e1 = e->R.e;
	if (!(R = (*e1->op)(e1)))
		zero_div(L, "/");
	rv = L / R;
	if (want_deriv)
		e->dR = -rv * (e->dL = 1. / R);
	return rv;
	}

 static real
#ifdef KR_headers
f_OPREM(e) register expr *e;
#else
f_OPREM(register expr *e)
#endif
{
	register expr *e1;
	real L, R, rv;
	/* e->dL = 1.; */
	e1 = e->L.e;
	L = (*e1->op)(e1);
	e1 = e->R.e;
	R = (*e1->op)(e1);
	rv = fmod(L,R);
	if (errno)
		introuble2("fmod",L,R);
	else
		e->dR = (rv - L) / R;
	return rv;
	}

 static real
#ifdef KR_headers
f_OPPOW(e) register expr *e;
#else
f_OPPOW(register expr *e)
#endif
{
	register expr *e1;
	real L, R, rv;
	e1 = e->L.e;
	L = (*e1->op)(e1);
	e1 = e->R.e;
	R = (*e1->op)(e1);
	rv = mypow(L,R);
	if (errno)
		introuble2("pow",L,R);
	if (want_deriv) {
		if (L > 0.) {
			e->dL = (R/L) * rv;
			e->dR = log(L) * rv;
			}
		else if (L != 0.) {
 bad:
			introuble2("pow'",L,R);
			}
		else {
			if (R > 1.)
				e->dL = e->dR = 0.;
			else if (R == 1.) {
				e->dL = 1.;
				e->dR = 0.;
				}
			else goto bad;
			}
		}
	return rv;
	}
 static real
#ifdef KR_headers
f_OP1POW(e) register expr *e;
#else
f_OP1POW(register expr *e)	/* f_OPPOW for R = numeric constant */
#endif
{
	register expr *e1;
	real L, R, rv;
	e1 = e->L.e;
	L = (*e1->op)(e1);
	R = ((expr_n *)e->R.e)->v;
	rv = mypow(L,R);
	if (errno)
		introuble2("pow",L,R);
	if (want_deriv) {
		if (L)
			e->dL = (R/L) * rv;
		else if (R > 1.)
			e->dL = 0.;
		else
			introuble2("pow'",L,R);
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_OP2POW(e) register expr *e;
#else
f_OP2POW(register expr *e)	/* f_OPPOW for R = 2 */
#endif
{
	register expr *e1;
	real L;
	e1 = e->L.e;
	L = (*e1->op)(e1);
	e->dL = L + L;
	return L*L;
	}

 static real
#ifdef KR_headers
f_OPCPOW(e) register expr *e;
#else
f_OPCPOW(register expr *e)	/* f_OPPOW for L = numeric constant */
#endif
{
	register expr *e1;
	real L, R, rv;
	e1 = e->R.e;
	rv = mypow(L = e->L.en->v, R = (*e1->op)(e1));
	if (errno)
		introuble2("pow",L,R);
	if (want_deriv) {
		if (L > 0) {
			if (e->dL == 1)
				e->dL = log(L);	/* cache value */
			e->dR = e->dL * rv;
			}
		else if (L == 0 && R >= 1.)
			e->dR = 0.;
		else
			introuble2("pow'",L,R);
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_OPLESS(e) register expr *e;
#else
f_OPLESS(register expr *e)
#endif
{
	register expr *e1;
	real L;
	e1 = e->L.e;
	L = (*e1->op)(e1);
	e1 = e->R.e;
	L -= (*e1->op)(e1);
	if (L < 0.)
		return e->dL = e->dR = 0;
	e->dL = 1.;
	e->dR = -1.;
	return L;
	}

 static real
#ifdef KR_headers
f_MINLIST(e0) expr *e0;
#else
f_MINLIST(expr *e0)
#endif
{
	de *d, *d1;
	real t, rv;
	expr *e1;
	derp *D;
	register expr_va *e = (expr_va *)e0;

	d = e->L.d;
	e1 = d->e;
	rv = (*e1->op)(e1);
	for(d1 = d++; e1 = d->e; d++) {
		t = (*e1->op)(e1);
		if (rv > t) {
			rv = t;
			d1 = d;
			}
		}
	if (D = e->R.D) {
		D->a.rp = d1->dv.rp;
		D->next = d1->d;
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_MAXLIST(e0) expr *e0;
#else
f_MAXLIST(expr *e0)
#endif
{
	de *d, *d1;
	real t, rv;
	expr *e1;
	derp *D;
	register expr_va *e = (expr_va *)e0;

	d = e->L.d;
	e1 = d->e;
	rv = (*e1->op)(e1);
	for(d1 = d++; e1 = d->e; d++) {
		t = (*e1->op)(e1);
		if (rv < t) {
			rv = t;
			d1 = d;
			}
		}
	if (D = e->R.D) {
		D->a.rp = d1->dv.rp;
		D->next = d1->d;
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_FLOOR(e) register expr *e;
#else
f_FLOOR(register expr *e)
#endif
{
	/* e->dL = 0.; */
	e = e->L.e;
	return floor((*e->op)(e));
	}

 static real
#ifdef KR_headers
f_CEIL(e) register expr *e;
#else
f_CEIL(register expr *e)
#endif
{
	/* e->dL = 0.; */
	e = e->L.e;
	return ceil((*e->op)(e));
	}

 static real
#ifdef KR_headers
f_ABS(e) register expr *e;
#else
f_ABS(register expr *e)
#endif
{
	real rv;
	register expr *e1;
	e1 = e->L.e;
	rv = (*e1->op)(e1);
	if (rv < 0.) {
		e->dL = -1.;
		return -rv;
		}
	e->dL = 1.;
	return rv;
	}

 static real
#ifdef KR_headers
f_OPUMINUS(e) register expr *e;
#else
f_OPUMINUS(register expr *e)
#endif
{
	/* e->dL = -1.; */
	e = e->L.e;
	return -(*e->op)(e);
	}

 static real
#ifdef KR_headers
f_OP_tanh(e) register expr *e;
#else
f_OP_tanh(register expr *e)
#endif
{
	real rv, t, t1;
	register expr *e1;
	e1 = e->L.e;
	rv = tanh(t = (*e1->op)(e1));
	if (errno)
		introuble("tanh",t);
	if (want_deriv) {
		t1 = 1. / cosh(t);
		if (errno)
			introuble("tanh'", t);
		e->dL = t1*t1;
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_tan(e) register expr *e;
#else
f_OP_tan(register expr *e)
#endif
{
	real rv, t, t1;
	register expr *e1;
	e1 = e->L.e;
	rv = tan(t = (*e1->op)(e1));
	if (errno)
		introuble("tan",t);
	if (want_deriv) {
		t1 = cos(t);
		if (errno || !t1)
			introuble("tan'",t);
		t1 = 1. / t1;
		e->dL = t1*t1;
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_sqrt(e) register expr *e;
#else
f_OP_sqrt(register expr *e)
#endif
{
	real t, rv;
	register expr *e1;
	e1 = e->L.e;
	rv = sqrt(t = (*e1->op)(e1));
	if (errno)
		introuble("sqrt",t);
	if (want_deriv) {
		if (rv <= 0.)
			introuble("sqrt'",t);
		e->dL = 0.5 / rv;
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_sinh(e) register expr *e;
#else
f_OP_sinh(register expr *e)
#endif
{
	real t, rv;
	register expr *e1;
	e1 = e->L.e;
	rv = sinh(t = (*e1->op)(e1));
	if (errno)
		introuble("sinh",t);
	if (want_deriv) {
		e->dL = cosh(t);
		if (errno)
			introuble("sinh'",t);
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_sin(e) register expr *e;
#else
f_OP_sin(register expr *e)
#endif
{
	real t, rv;
	register expr *e1;
	e1 = e->L.e;
	rv = sin(t = (*e1->op)(e1));
	if (errno)
		introuble("sin",t);
	if (want_deriv) {
		e->dL = cos(t);
		if (errno)
			introuble("sin'",t);
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_log10(e) register expr *e;
#else
f_OP_log10(register expr *e)
#endif
{
	real t, rv;
	register expr *e1;
	static real Le10;

	e1 = e->L.e;
	rv = log10(t = (*e1->op)(e1));
	if (errno)
		introuble("log10",t);
	if (want_deriv) {
		if (!Le10)
			Le10 = 1. / log(10.);
		e->dL = Le10 / t;
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_log(e) register expr *e;
#else
f_OP_log(register expr *e)
#endif
{
	real t, rv;
	register expr *e1;
	e1 = e->L.e;
	rv = log(t = (*e1->op)(e1));
	if (errno)
		introuble("log",t);
	if (want_deriv)
		e->dL = 1. / t;
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_exp(e) register expr *e;
#else
f_OP_exp(register expr *e)
#endif
{
	real t, rv;
	register expr *e1;
	e1 = e->L.e;
	rv = e->dL = exp(t = (*e1->op)(e1));
	if (errno)
		if (t >= 0.)
			introuble("exp",t);
		else {
			errno = 0;
			rv = 0.;
			}
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_cosh(e) register expr *e;
#else
f_OP_cosh(register expr *e)
#endif
{
	real t, rv;
	register expr *e1;
	e1 = e->L.e;
	rv = cosh(t = (*e1->op)(e1));
	if (errno)
		introuble("cosh",t);
	if (want_deriv) {
		e->dL = sinh(t);
		if (errno)
			introuble("cosh'",t);
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_cos(e) register expr *e;
#else
f_OP_cos(register expr *e)
#endif
{
	real t, rv;
	register expr *e1;
	e1 = e->L.e;
	rv = cos(t = (*e1->op)(e1));
	if (errno)
		introuble("cos",t);
	if (want_deriv) {
		e->dL = -sin(t);
		if (errno)
			introuble("cos'",t);
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_atanh(e) register expr *e;
#else
f_OP_atanh(register expr *e)
#endif
{
	real t, rv;
	register expr *e1;

	e1 = e->L.e;
	t = (*e1->op)(e1);
	if (t <= -1. || t >= 1.) {
		errno = EDOM;
		rv = 0.;
		}
	else
		rv = 0.5*log((1. + t) / (1. - t));
	if (errno)
		introuble("atanh",t);
	if (want_deriv)
		e->dL = 1. / (1. - t*t);
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_atan2(e) register expr *e;
#else
f_OP_atan2(register expr *e)
#endif
{
	real L, R, rv, t, t1;
	register expr *e1;

	e1 = e->L.e;
	L = (*e1->op)(e1);
	e1 = e->R.e;
	R = (*e1->op)(e1);
	rv = atan2(L,R);
	if (errno)
		introuble2("atan2",L,R);
	if (want_deriv) {
		if ((t = L) < 0.)
			t = -t;
		if ((t1 = R) < 0.)
			t1 = -t1;
		if (t1 >= t) {
			t = L / R;
			t1 = 1. / (1. + t*t);
			e->dL = t1 /= R;
			e->dR = -t*t1;
			}
		else {
			t = R / L;
			t1 = -1. / (1. + t*t);
			e->dR = t1 /= L;
			e->dL = -t*t1;
			}
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_atan(e) register expr *e;
#else
f_OP_atan(register expr *e)
#endif
{
	real t, rv;
	register expr *e1;

	e1 = e->L.e;
	rv = atan(t = (*e1->op)(e1));
	if (errno)
		introuble("atan",t);
	if (want_deriv)
		e->dL = 1. / (1. + t*t);
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_asinh(e) register expr *e;
#else
f_OP_asinh(register expr *e)
#endif
{
	register expr *e1;
	real rv, t, t0, t1;
	int sign;

	e1 = e->L.e;
	t = t0 = (*e1->op)(e1);
	if (sign = t < 0.)
		t = -t;
	rv = log(t + (t1 = sqrt(t*t + 1.)));
	if (sign)
		rv = -rv;
	if (errno)
		introuble("asinh",t0);
	if (want_deriv)
		e->dL = 1. / t1;
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_asin(e) register expr *e;
#else
f_OP_asin(register expr *e)
#endif
{
	register expr *e1;
	real rv, t, t1;

	e1 = e->L.e;
	rv = asin(t = (*e1->op)(e1));
	if (errno)
		introuble("asin",t);
	if (want_deriv) {
		if ((t1 = 1. - t*t) <= 0.)
			introuble("asin'",t);
		e->dL = 1. / sqrt(t1);
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_acosh(e) register expr *e;
#else
f_OP_acosh(register expr *e)
#endif
{
	register expr *e1;
	real rv, t, t1;

	e1 = e->L.e;
	t = (*e1->op)(e1);
	if (t < 1.) {
		errno = EDOM;
		rv = 0.;
		}
	else
		rv = log(t + (t1 = sqrt(t*t - 1.)));
	if (errno)
		introuble("acosh",t);
	if (want_deriv)
		e->dL = 1. / t1;
	return rv;
	}

 static real
#ifdef KR_headers
f_OP_acos(e) register expr *e;
#else
f_OP_acos(register expr *e)
#endif
{
	register expr *e1;
	real rv, t, t1;

	e1 = e->L.e;
	rv = acos(t = (*e1->op)(e1));
	if (errno)
		introuble("acos",t);
	if (want_deriv) {
		if ((t1 = 1. - t*t) <= 0.)
			introuble("acos'",t);
		e->dL = -1. / sqrt(t1);
		}
	return rv;
	}

 static real
#ifdef KR_headers
f_OPIFnl(e) register expr *e;
#else
f_OPIFnl(register expr *e)
#endif
{
	register expr_if *eif = (expr_if *)e;
	register derp *D;

	e = eif->e;
	if ((*e->op)(e)) {
		e = eif->T;
		if (D = eif->D) {
			D->a.rp = eif->Tv.rp;
			D->next = eif->dT;
			}
		}
	else {
		e = eif->F;
		if (D = eif->D) {
			D->a.rp = eif->Fv.rp;
			D->next = eif->dF;
			}
		}
	return (*e->op)(e);
	}

 static real
#ifdef KR_headers
f_OPOR(e) register expr *e;
#else
f_OPOR(register expr *e)
#endif
{
	register expr *e2;
	e2 = e->R.e;
	e = e->L.e;
	return (*e->op)(e) || (*e2->op)(e) ? 1. : 0.;
	}

 static real
#ifdef KR_headers
f_OPAND(e) register expr *e;
#else
f_OPAND(register expr *e)
#endif
{
	register expr *e2;
	e2 = e->R.e;
	e = e->L.e;
	return (*e->op)(e) && (*e2->op)(e) ? 1. : 0.;
	}

 static real
#ifdef KR_headers
f_LT(e) register expr *e;
#else
f_LT(register expr *e)
#endif
{
	register expr *e2;
	e2 = e->R.e;
	e = e->L.e;
	return (*e->op)(e) < (*e2->op)(e2) ? 1. : 0;
	}

 static real
#ifdef KR_headers
f_LE(e) register expr *e;
#else
f_LE(register expr *e)
#endif
{
	register expr *e2;
	e2 = e->R.e;
	e = e->L.e;
	return (*e->op)(e) <= (*e2->op)(e2) ? 1. : 0;
	}

 static real
#ifdef KR_headers
f_EQ(e) register expr *e;
#else
f_EQ(register expr *e)
#endif
{
	register expr *e2;
	e2 = e->R.e;
	e = e->L.e;
	return (*e->op)(e) == (*e2->op)(e2) ? 1. : 0;
	}

 static real
#ifdef KR_headers
f_GE(e) register expr *e;
#else
f_GE(register expr *e)
#endif
{
	register expr *e2;
	e2 = e->R.e;
	e = e->L.e;
	return (*e->op)(e) >= (*e2->op)(e2) ? 1. : 0;
	}

 static real
#ifdef KR_headers
f_GT(e) register expr *e;
#else
f_GT(register expr *e)
#endif
{
	register expr *e2;
	e2 = e->R.e;
	e = e->L.e;
	return (*e->op)(e) > (*e2->op)(e2) ? 1. : 0;
	}

 static real
#ifdef KR_headers
f_NE(e) register expr *e;
#else
f_NE(register expr *e)
#endif
{
	register expr *e2;
	e2 = e->R.e;
	e = e->L.e;
	return (*e->op)(e) != (*e2->op)(e2) ? 1. : 0;
	}

 static real
#ifdef KR_headers
f_OPNOT(e) register expr *e;
#else
f_OPNOT(register expr *e)
#endif
{
	e = e->L.e;
	return (*e->op)(e) ? 0. : 1.;
	}

 static real
#ifdef KR_headers
f_OPSUMLIST(e) expr *e;
#else
f_OPSUMLIST(expr *e)
#endif
{
	register expr **ep, **epe;
	real x;
	ep = e->L.ep;
	epe = e->R.ep;
	e = *ep++;
	x = (*e->op)(e);
	do {
		e = *ep++;
		x += (*e->op)(e);
		}
		while(ep < epe);
	return x;
	}

 static real
#ifdef KR_headers
f_OPintDIV(e) expr *e;
#else
f_OPintDIV(expr *e)
#endif
{
	register expr *e1;
	real L, R;

	e1 = e->L.e;
	L = (*e1->op)(e1);
	e1 = e->R.e;
	if (!(R = (*e1->op)(e1)))
		zero_div(L, " div ");
	return (L /= R) >= 0 ? floor(L) : ceil(L);
	}

 static real
#ifdef KR_headers
f_OPprecision(e) expr *e;
#else
f_OPprecision(expr *e)
#endif
{
	register expr *e1;
	real L, R;
	char buf[32];

	e1 = e->L.e;
	L = (*e1->op)(e1);
	e1 = e->R.e;
	R = (*e1->op)(e1);
	g_fmtp(buf, L, (int)R);
	return strtod(buf, (char **)0);
	}

#ifdef No_dtoa

 static real
#ifdef KR_headers
round(x, prec) real x; int prec;
#else
round(real x, int prec)
#endif
{
	real scale;
	int flip;

	if (!x)
		return x;
	flip = 0;
	if (x < 0.) {
		x = -x;
		flip = 1;
		}
	if (!prec)
		x = floor(x + 0.5);
	else if (prec > 0) {
		scale = mypow(10., (real)prec);
		x = floor(x*scale + 0.5) / scale;
		}
	else {
		scale = mypow(10., -(real)prec);
		x = scale*floor(x/scale + 0.5);
		}
	return flip ? -x : x;
	}
#else

#ifdef KR_headers
extern char *dtoa();
#else
extern char *dtoa(double, int, int, int*, int*, char **);
#endif

 static real
#ifdef KR_headers
round(x, prec) real x; int prec;
#else
round(real x, int prec)
#endif
{
	char *b, *s, *se;
	int decpt, L, sign;
	char buf[96];

	if (!x)
		return x;
	s = dtoa(x, 3, prec, &decpt, &sign, &se);
	if (decpt == 9999)
		return x;
	L = se - s;
	if (L <= 0)
		return 0;
	if (L > 80)
		se = s + 80;
	b = buf;
	if (sign)
	*b++ = '-';
	*b++ = '.';
	while(s < se)
		*b++ = *s++;
	*b = 0;
	if (decpt)
		sprintf(b, "e%d", decpt);
	return strtod(buf, (char **)0);
	}
#endif

 static real
#ifdef KR_headers
f_OPround(e) expr *e;
#else
f_OPround(expr *e)
#endif
{
	register expr *e1;
	real L, R;

	e1 = e->L.e;
	L = (*e1->op)(e1);
	e1 = e->R.e;
	R = (*e1->op)(e1);
	return round(L, (int)R);
	}

 static real
#ifdef KR_headers
f_OPtrunc(e) expr *e;
#else
f_OPtrunc(expr *e)
#endif
{
	register expr *e1;
	real L, R;
	int prec;

	e1 = e->L.e;
	L = (*e1->op)(e1);
	e1 = e->R.e;
	if (!(R = (*e1->op)(e1)))
		return L >= 0. ? floor(L) : ceil(L);
	R = round(L, prec = (int)R);
	if (R != L) {
		R = 0.5*mypow(10., (real)-prec);
		R = round(L > 0 ? L - R : L + R, prec);
		}
	return R;
	}

 static real
#ifdef KR_headers
f_OPPLTERM(e) expr *e;
#else
f_OPPLTERM(expr *e)
#endif
{
	plterm *p = e->L.p;
	real r, t;
	int n = p->n;
	real *bs = p->bs;

	r = ((expr_v *)e->R.e)->v;
	if (r >= 0) {
		/* should use binary search for large n */
		while(bs[1] <= 0.) {
			bs += 2;
			if (--n <= 1)
				return r*(e->dL = *bs);
			}
		if (r <= bs[1])
			return r*(e->dL = *bs);
		for(t = bs[0]*bs[1]; --n > 1 && r > bs[3]; bs += 2)
			t += (bs[3]-bs[1])*bs[2];
		return t + (r-bs[1])*(e->dL = bs[2]);
		}
	bs += 2*(n-1);
	while(bs[-1] >= 0.) {
		bs -= 2;
		if (--n <= 1)
			return r*(e->dL = *bs);
		}
	if (r >= bs[-1])
		return r*(e->dL = *bs);
	for(t = bs[0]*bs[-1]; --n > 1 && r < bs[-3]; bs -= 2)
		t += (bs[-3]-bs[-1])*bs[-2];
	return t + (r-bs[-1])*(e->dL = bs[-2]);
	}

 real
#ifdef KR_headers
f_OPNUM(e) expr *e;
#else
f_OPNUM(expr *e)
#endif
{ return ((expr_n *)e)->v; }

typedef char * sfunc ANSI((expr*));

 static char *
#ifdef KR_headers
f_OPHOL(e) expr *e;
#else
f_OPHOL(expr *e)
#endif
{ return ((expr_h *)e)->sym; }

 static real
#ifdef KR_headers
f_OPVARVAL(e) expr *e;
#else
f_OPVARVAL(expr *e)
#endif
{ return ((expr_v *)e)->v; }

 static char *
#ifdef KR_headers
f_OPIFSYM(e) register expr *e;
#else
f_OPIFSYM(register expr *e)
#endif
{
	register expr_if *eif = (expr_if *)e;
	e = eif->e;
	e = (*e->op)(e) ? eif->T : eif->F;
	return (*(sfunc*)e->op)(e);
	}

 static real
#ifdef KR_headers
f_OPFUNCALL(e) register expr *e;
#else
f_OPFUNCALL(register expr *e)
#endif
{
	register expr_f *f = (expr_f *)e;
	register argpair *ap, *ape;
	for(ap = f->ap, ape = f->ape; ap < ape; ap++) {
		e = ap->e;
		*ap->u.v = (*e->op)(e);
		}
	for(ap = f->sap, ape = f->sape; ap < ape; ap++) {
		e = ap->e;
		*ap->u.s = (*(sfunc*)e->op)(e);
		}
	return (*f->f)(f->al);
	}

 efunc *
r_ops[] = {
#include "r_op.hd"
	};
