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
#include "r_opn.hd"	/* for f_OPNUM */
extern efunc *r_ops[];
extern char *progname;

#include "r_qp.hd"

#define GULP		200

 static fint *s_s, *s_z;
 static double *s_x;

 typedef struct
dyad {
	struct dyad *next;
	ograd *Lq, *Rq;
	} dyad;

 typedef struct
term {
	dyad	*Q, *Qe;
	ograd	*L, *Le;
	} term;

 static void
#ifdef KR_headers
nonlin()
#else
nonlin(void)
#endif
{
	fprintf(stderr,
		"Sorry, %s can't handle nonlinearities.\n",
		progname ? progname : "");
	exit(1);
	}

static term *freeterm;
static ograd *freeog;
static dyad *freedyad;
static int zerodiv;
static term **cterms;

 static void
#ifdef KR_headers
free_term(t) term *t;
#else
free_term(term *t)
#endif
{
	t->Q = (dyad *)freeterm;
	freeterm = t;
	}

 static term *
#ifdef KR_headers
new_term(o) ograd *o;
#else
new_term(ograd *o)
#endif
{
	static int ntogo;
	static term *block;
	term *rv;

	if (rv = freeterm)
		freeterm = (term *)rv->Q;
	else {
		if (!ntogo) {
			block = (term *)Malloc(GULP*sizeof(term));
			ntogo = GULP;
			}
		rv = block++;
		--ntogo;
		}
	rv->L = rv->Le = o;
	rv->Q = rv->Qe = 0;
	return rv;
	}

 static void
#ifdef KR_headers
free_og(o) ograd *o;
#else
free_og(ograd *o)
#endif
{
	o->next = freeog;
	freeog = o;
	}

 static ograd *
#ifdef KR_headers
new_og(next, i, v) ograd *next; int i; real v;
#else
new_og(ograd *next, int i, real v)
#endif
{
	static int ntogo;
	static ograd *block;
	ograd *rv;

	if (rv = freeog)
		freeog = rv->next;
	else {
		if (!ntogo) {
			block = (ograd *)Malloc(GULP*sizeof(ograd));
			ntogo = GULP;
			}
		rv = block++;
		--ntogo;
		}
	rv->next = next;
	rv->varno = i;
	rv->coef = v;
	return rv;
	}

 static ograd *
#ifdef KR_headers
ogdup(og, oge) ograd *og, **oge;
#else
ogdup(ograd *og, ograd **oge)
#endif
{
	ograd *og0, *og1;

	og0 = og1 = new_og(0, og->varno, og->coef);
	while(og = og->next)
		og1 = og1->next = new_og(0, og->varno, og->coef);
	if (oge)
		*oge = og1;
	return og0;
	}

 static int
#ifdef KR_headers
count(ogp) ograd **ogp;
#else
count(ograd **ogp)
#endif
{
	int i, rv, nz;
	fint *s, *z;
	double t, *x;
	ograd *og, *og1=NULL;

	s = s_s;
	x = s_x;
	z = s_z;

	t = 0;
	nz = rv = 0;
	for(og = *ogp; og; og = og1) {
		og1 = og->next;
		if ((i = og->varno) < 0)
			t += og->coef;
		else if (!s[i]++)
			x[z[nz++] = i] = og->coef;
		else
			x[i] += og->coef;
		free_og(og);
		}
	while(nz > 0) {
		s[i = z[--nz]] = 0;
		if (x[i]) {
			og = new_og(og, i, x[i]);
			rv++;
			}
		}
	if (t)
		og = new_og(og, -1, t);
	*ogp = og;
	return rv;
	}

 static void
#ifdef KR_headers
free_dyad(t) dyad *t;
#else
free_dyad(dyad *t)
#endif
{
	t->next = freedyad;
	freedyad = t;
	}

 static dyad *
#ifdef KR_headers
new_dyad(next, L, R, permute) dyad *next; ograd *L, *R; int permute;
#else
new_dyad(dyad *next, ograd *L, ograd *R, int permute)
#endif
{
	static int ntogo;
	static dyad *block;
	dyad *rv;
	ograd *t;

	if (permute) {
		if (L == R) {
			count(&L);
			R = L;
			}
		else if (count(&L) > count(&R)) {
			t = L;
			L = R;
			R = t;
			}
		}
	if (rv = freedyad)
		freedyad = rv->next;
	else {
		if (!ntogo) {
			block = (dyad *)Malloc(GULP*sizeof(dyad));
			ntogo = GULP;
			}
		rv = block++;
		--ntogo;
		}
	rv->next = next;
	rv->Lq = L;
	rv->Rq = R;
	return rv;
	}

 static term *
#ifdef KR_headers
termsum(L, R) term *L, *R;
#else
termsum(term *L, term *R)
#endif
{
	if (L->Qe && (L->Qe->next = R->Q))
		L->Qe = R->Qe;
	else if (R->Q) {
		L->Q = R->Q;
		L->Qe = R->Qe;
		}
	if (L->Le && (L->Le->next = R->L))
		L->Le = R->Le;
	else if (R->L) {
		L->L = R->L;
		L->Le = R->Le;
		}
	free_term(R);
	return L;
	}

 static term *
#ifdef KR_headers
scale(T, t) term *T; register double t;
#else
scale(term *T, register double t)
#endif
{
	register ograd *og;
	register dyad *d;

	for(d = T->Q; d; d = d->next) {
		if (d->Lq == d->Rq)
			d->Rq = ogdup(d->Lq, 0);
		for(og = d->Lq; og; og = og->next)
			og->coef *= t;
		}
	for(og = T->L; og; og = og->next)
		og->coef *= t;
	return T;
	}

 static term *ewalk(
#ifndef KR_headers
		    expr*
#endif
			 );

 static term *
#ifdef KR_headers
comterm(i) int i;
#else
comterm(int i)
#endif
{
	int nlin;
	cexp *c;
	cexp1 *c1;
	linpart *L, *Le;
	expr_v ev, *vp;
	term *T;

	if (i < ncom0) {
		c = cexps + i;
		T = ewalk(c->e);
		L = c->L;
		nlin = c->nlin;
		}
	else {
		c1 = cexps1 + (i - ncom0);
		T = ewalk(c1->e);
		L = c1->L;
		nlin = c1->nlin;
		}
	if (L)
		for(Le = L + nlin; L < Le; L++) {
			vp = (expr_v *)((char *)L->v.rp -
				((char *)&ev.v - (char *)&ev));
			T = termsum(T, new_term(
				new_og(0, (int)(vp - var_e), L->fac)));
			}
	return T;
	}

 static term *
#ifdef KR_headers
termdup(T) term *T;
#else
termdup(term *T)
#endif
{
	term *rv;
	ograd *og, *oge;
	dyad *Q, *Q1;

	if (og = oge = T->L)
		og = ogdup(og, &oge);
	rv = new_term(og);
	rv->Le = oge;
	if (!(Q = T->Q))
		return rv;
	for(Q1 = 0; Q; Q = Q->next)
		Q1 = new_dyad(Q1, ogdup(Q->Lq,0), ogdup(Q->Rq,0), 1);
	rv->Qe = Q1;
	while(Q = Q1->next)
		Q1 = Q;
	rv->Q = Q1;
	return rv;
	}

 static term *
#ifdef KR_headers
ewalk(e) expr *e;
#else
ewalk(expr *e)
#endif
{
	term *L, *R, *T;
	ograd *o, *or;
	expr **ep, **epe;
	int i;

	switch((int)e->op) {

	  case OPNUM:
		return new_term(new_og(0, -1 , ((expr_n *)e)->v));

	  case OPPLUS:
		return termsum(ewalk(e->L.e), ewalk(e->R.e));

	  case OPMINUS:
		return termsum(ewalk(e->L.e), scale(ewalk(e->R.e), -1.));

	  case OPUMINUS:
		return scale(ewalk(e->L.e), -1.);

	  case OPMULT:
		L = ewalk(e->L.e);
		R = ewalk(e->R.e);
		if (L->Q) {
			if (R->Q)
				nonlin();
 qscale:
			o = R->L;
			if (o->next || o->varno >= 0)
				nonlin();
			scale(L, o->coef);
			free_og(o);
			free_term(R);
			return L;
			}
		if (R->Q) {
			T = L;
			L = R;
			R = T;
			goto qscale;
			}
		o = L->L;
		or = R->L;
		if (o->next || o->varno >= 0) {
			if (or->next || or->varno >= 0) {
				L->Q = L->Qe = new_dyad(0,o,or,1);
				L->L = L->Le = 0;
				}
			else {
				scale(L, or->coef);
				free_og(or);
				}
			free_term(R);
			return L;
			}
		scale(R, o->coef);
		free_og(o);
		free_term(L);
		return R;

	  case OPDIV:
		/* only allow division by a constant */
		R = ewalk(e->R.e);
		o = R->L;
		if (R->Q || o->next || o->varno >= 0)
			nonlin();
		L = ewalk(e->L.e);
		if (!o->coef)
			zerodiv++;
		else
			scale(L, 1./o->coef);
		free_og(o);
		free_term(R);
		return L;

	  case OPSUMLIST:
		ep = e->L.ep;
		epe = e->R.ep;
		L = ewalk(*ep);
		while(++ep < epe)
			termsum(L, ewalk(*ep));
		return L;

	  case OP2POW:
		L = ewalk(e->L.e);
		if (L->Q)
			nonlin();
		o = L->L;
		if (!o->next && o->varno < 0) {
			o->coef *= o->coef;
			return L;
			}
		L->Q = L->Qe = new_dyad(0,o,o,1);
		L->L = L->Le = 0;
		return L;

	  case OPVARVAL:
		if ((i = (expr_v *)e - var_e) < n_var)
			return new_term(new_og(0, i, 1.));
		if (!(L = cterms[i -= n_var]))
			L = cterms[i] = comterm(i);
		return termdup(L);

	  default:
		nonlin();
		}
	/* NOT REACHED */ return 0;
	}

 static int
#ifdef KR_headers
comp(a, b) char *a, *b;
#else
comp(const void *a, const void *b)
#endif
{ return (*(ograd **)a)->varno - (*(ograd **)b)->varno; }

 static ograd *
#ifdef KR_headers
sortq(og0, q) ograd *og0, **q;
#else
sortq(ograd *og0, ograd **q)
#endif
{
	ograd *og, **q1;
	int n;

	for(q1 = q, og = og0; og; og = og->next)
		*q1++ = og;
	if ((n = q1 - q) > 1) {
		qsort(q, n, sizeof(ograd *), comp);
		og0 = 0;
		do {
			og = *--q1;
			og->next = og0;
			og0 = og;
			} while(q1 > q);
		}
	return og0;
	}

 static double
#ifdef KR_headers
dsort(T, q) term *T; ograd **q;
#else
dsort(term *T, ograd **q)
#endif
{
	ograd *og, *og1;
	double t, t1, rv, *x;
	dyad *Q;

	x = s_x;

	rv = 0;
	count(&T->L);	/* accumulate */
	for(og = Ograd[obj_no]; og; og = og->next)
		x[og->varno] = og->coef;
	if ((og = T->L) && og->varno < 0) {
		rv = og->coef;
		og = og->next;
		}
	for(; og; og = og->next)
		x[og->varno] += og->coef;

	for(Q = T->Q; Q; Q = Q->next) {
		og = Q->Lq;
		og1 = Q->Rq;
		t = t1 = 0;
		if (og->varno < 0) {
			t = og->coef;
			og = og->next;
			}
		if (og1->varno < 0) {
			t1 = og1->coef;
			Q->Rq = og1 = og1->next;
			rv += t*t1;
			}
		if (t)
			for(; og1; og1 = og1->next)
				x[og1->varno] += t*og1->coef;
		if (t1)
			for(og1 = og; og1; og1 = og1->next)
				x[og1->varno] += t1*og1->coef;
		Q->Lq = sortq(og, q);
		Q->Rq = og == Q->Rq ? Q->Lq : sortq(Q->Rq, q);
		}
	for(og = Ograd[obj_no]; og; og = og->next)
		og->coef = x[og->varno];
	return rv;
	}

 static void
#ifdef KR_headers
free_oglist(og) ograd *og;
#else
free_oglist(ograd *og)
#endif
{
	ograd *og1=NULL;

	for(; og; og = og1) {
		og1 = og->next;
		free_og(og);
		}
	}

 static void
#ifdef KR_headers
cterm_free(cte) term **cte;
#else
cterm_free(term **cte)
#endif
{
	term **ct, *t;
	dyad *d, *d1;

	for(ct = cterms; ct < cte; ct++)
		if (t = *ct) {
			free_oglist(t->L);
			d1 = t->Q;
			while(d = d1) {
				d1 = d->next;
				free_oglist(d->Lq);
				if (d->Rq != d->Lq)
					free_oglist(d->Rq);
				free_dyad(d);
				}
			}
	free((char *)cterms);
	}

 static int
#ifdef KR_headers
lcmp(a, b) char *a, *b;
#else
lcmp(const void *a, const void *b)
#endif
{ return (int)(*(fint *)a - *(fint *)b); }

 fint
#ifdef KR_headers
qpcheck(rowqp, colqp, delsqp) fint **rowqp,**colqp;double **delsqp;
#else
qpcheck(fint **rowqp, fint **colqp, double **delsqp)
#endif
{
	expr *e;
	term *T;
	double *delsq, *delsq0, objadj, t, *x;
	int pass;
	fint ftn, i, icol, j, ncom, nelq, nz;
	fint *colq, *rowq, *rowq0, *s, *z;
	dyad *d, *d1, **q, **q1, **q2, **qe;
	ograd *og, *og1, *og2;
	expr_n *en;
	typedef struct dispatch {
		struct dispatch *next;
		fint i, j, jend;
		} dispatch;
	dispatch *cd, *cd0, **cdisp, **cdisp0, *cdnext, **cdp;

	if (obj_no < 0 || obj_no >= n_obj)
		return 0;
	e = obj_de[obj_no].e;
	if (e->op == f_OPNUM)
		return 0;

	s_x = x = (double *)Malloc(n_var*(sizeof(double)+2*sizeof(fint)));
	s_z = z = (fint *)(x + n_var);
	s_s = s = z + n_var;
	memset((char *)s, 0, n_var*sizeof(fint));
	ftn = Fortran;

	if (comb + como + como1) {
		ncom = ncom0 + ncom1;
		cterms = (term **)Malloc(ncom*sizeof(term*));
		memset((char *)cterms, 0, ncom*sizeof(term*));
		}

	T = ewalk(e);
	if (zerodiv) {
		fprintf(stderr,
			"Quadratic objective involves division by 0.\n");
		exit(1);
		}

	if (cterms)
		cterm_free(cterms + ncom);

	q = (dyad **)Malloc(n_var*sizeof(dyad *));
	qe = q + n_var;
	objadj = dsort(T, (ograd **)q);

	nelq = nz = 0;
	for(pass = 0; pass < 2; pass++) {
		if (pass) {
			free((char *)q);
			*delsqp = delsq = (double *)Malloc(nelq*sizeof(real));
			*rowqp = rowq = (fint *)Malloc(nelq*sizeof(fint));
			*colqp = colq = (fint *)Malloc((n_var+2)*sizeof(fint));
			nelq = ftn;
			delsq0 = delsq - ftn;
			rowq0 = rowq - ftn;
			q = (dyad **)Malloc(n_var*(sizeof(dyad*)
						+ sizeof(dispatch *)
						+ sizeof(dispatch)));
			qe = q + n_var;
			cdisp = (dispatch**) qe;
			cdisp0 = cdisp - ftn;
			memset((char *)cdisp, 0,
				n_var*sizeof(dispatch*));
			cd0 = (dispatch *)(cdisp + n_var);
			}
		memset((char *)q, 0, n_var*sizeof(dyad *));

		if (pass)
			for(d = T->Q; d; d = d->next) {
				og = d->Rq;
				og1 = d->Lq;
				i = og->varno;
				while(og1 && og1->varno < i)
					og1 = og1->next;
				if (og1) {
					q1 = q + i;
					*q1 = new_dyad(*q1, og, og1, 0);
					}
				og1 = d->Lq;
				i = og1->varno;
				while(og && og->varno < i)
					og = og->next;
				if (og) {
					q1 = q + i;
					*q1 = new_dyad(*q1, og1, og, 0);
					}
				}
		else
			for(d = T->Q; d; d = d->next) {
				og = d->Rq;
				og1 = d->Lq;
				q1 = q + og->varno;
				*q1 = new_dyad(*q1, og, og1, 0);
				q1 = q + og1->varno;
				*q1 = new_dyad(*q1, og1, og, 0);
				}
		icol = -1;
		for(q1 = q; q1 < qe; q1++) {
		    if (pass) {
			*colq++ = nelq;
			for(cd = cdisp[++icol]; cd; cd = cdnext) {
				cdnext = cd->next;
				s[i = cd->i]++;
				x[z[nz++] = i] = delsq0[cd->j++];
				if (cd->j < cd->jend) {
					cdp = cdisp0 + rowq0[cd->j];
					cd->next = *cdp;
					*cdp = cd;
					}
				}
			}
		    if (d = *q1)
			do {
				og = d->Lq;
				og1 = d->Rq;
				switch(pass) {
				  case 0:
					for(; og1; og1 = og1->next)
						if (!s[i = og1->varno]++)
							z[nz++] = i;
					break;
				  case 1:
					t = og->coef;
					for(; og1; og1 = og1->next) {
						if (!s[i = og1->varno]++)
							x[z[nz++] = i] =
								t*og1->coef;
						else
							x[i] += t*og1->coef;
						}
					if (og1 = og->next) {
					  og2 = d->Rq;
					  while (og2->varno < og1->varno)
					    if (!(og2 = og2->next)) {
						while(og1 = og->next)
							og = og1;
						break;
						}
					  d->Rq = og2;
					  }
					}
				d1 = d->next;
				if (og = og->next) {
					i = og->varno;
					if (pass) {
						og1 = d->Rq;
						while(og1->varno < i)
							if (!(og1 = og1->next))
								goto d_del;
						d->Rq = og1;
						}
					d->Lq = og;
					q2 = q + i;
					d->next = *q2;
					*q2 = d;
					}
				else {
 d_del:
					free_dyad(d);
					}
				}
				while(d = d1);
		if (nz) {
			if (pass) {
				if (nz > 1)
					qsort((char*)z,nz,sizeof(fint),lcmp);
				for(i = 0; i < nz; i++) {
					if (t = x[j = z[i]]) {
						*delsq++ = t;
						*rowq++ = j + ftn;
						nelq++;
						}
					s[j] = 0;
					}
				for(i = 0; i < nz; i++)
				    if ((j = z[i]) > icol && x[j]) {
					cd0->i = icol;
					cd0->j = colq[-1] + i;
					cd0->jend = nelq;
					cdp = cdisp + j;
					cd0->next = *cdp;
					*cdp = cd0++;
					break;
					}
				nz = 0;
				}
			else {
				nelq += nz;
				while(nz > 0)
					s[z[--nz]] = 0;
				}
			}
		    }
		}
	free((char *)q);
	free((char *)x);
	*colq = colq[1] = nelq;	/* allow one more for obj. adjustment */
	en = (expr_n *)new_og(0,0,0);
	en->op = f_OPNUM;
	en->v = objadj;
	obj_de[obj_no].e = (expr *)en;
	return nelq - ftn;
	}
