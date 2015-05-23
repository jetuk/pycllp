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

 void
#ifdef KR_headers
derprop(d) register derp *d;
#else
derprop(register derp *d)
#endif
{
	if (d) {
		*d->b.rp = 1.;
		do *d->a.rp += *d->b.rp * *d->c.rp;
			while(d = d->next);
		}
	}

 void
#ifdef KR_headers
comeval(i, ie) int i, ie;
#else
comeval(int i, int ie)
#endif
{
	register cexp *c, *ce;
	register expr *e;
	register expr_v *V = var_ex + i;
	register linpart *L, *Le;
	real t;
	c = cexps + i;
	ce = cexps + ie;
	do {
		cv_index = ++i;	/* identify var in case of error */
		e = c->e;
		t = (*e->op)(e);
		if (L = c->L)
			for(Le = L + c->nlin; L < Le; L++)
				t += L->fac * *L->v.rp;
		(*V++).v = t;
		}
		while(++c < ce);
	cv_index = 0;
	}

 void
#ifdef KR_headers
com1eval(i, ie) int i, ie;
#else
com1eval(int i, int ie)
#endif

{
	register cexp1 *c, *ce;
	register expr *e;
	register expr_v *V = var_ex1 + i;
	register linpart *L, *Le;
	real t;
	c = cexps1 + i;
	ce = cexps1 + ie;
	i += ncom0;
	do {
		cv_index = ++i;	/* identify var in case of error */
		e = c->e;
		t = (*e->op)(e);
		if (L = c->L)
			for(Le = L + c->nlin; L < Le; L++)
				t += L->fac * *L->v.rp;
		(*V++).v = t;
		}
		while(++c < ce);
	cv_index = 0;
	}

 void
#ifdef KR_headers
funnelset(f) register funnel *f;
#else
funnelset(register funnel *f)
#endif
{
	register derp	*d;
	register cplist	*cl;

	for(; f; f = f->next) {
		memset((char *)adjoints_nv1, 0, f->fcde.zaplen);
		cl = f->cl;
		do *cl->ca.rp = 0;
			while(cl = cl->next);
		d = f->fcde.d;
		*d->b.rp = 1.;
		do *d->a.rp += *d->b.rp * *d->c.rp;
			while(d = d->next);
		cl = f->cl;
		do *cl->cfa = *cl->ca.rp;
			while(cl = cl->next);
		}
	}

 void
report_where(VOID)
{
	int i, j, k, k1;
	static char *what[2] = { "constraint", "objective" };
	static char *nfmt[2] = { "%d: ", "function: " };
	char *b, buf[512];
	FILE *f;

	fflush(stdout);
	need_nl = 0;
	fprintf(stderr, "Error evaluating ");

#define next_line fgets(buf,sizeof(buf),f)

	if (i = cv_index) {
		strcpy(stub_end, ".fix");
		j = 0;
		if (f = fopen(filename, "r")) {
			for(;;) {
				if (!next_line)
					goto eof;
				for(b = buf; *b; b++)
					if (*b == '=') {
						while(++j < i)
							if (!next_line)
								goto eof;
						b = buf;
						while(*b && *b != '=')
							b++;
						if (*b != '=' || b < buf + 2)
							j = 0;
						else
							b[-1] = 0;
						goto eof;
						}
				}
 eof:
			fclose(f);
			}
		if (j == i)
			fprintf(stderr, "var %s: ", buf);
		else
			fprintf(stderr, "\"var =\" definition %d: ", i);
		goto ret;
		}

	k = k1 = 0;
	if ((i = co_index) < 0) {
		k = 1;
		i = n_con -i - 1;
		if (n_obj <= 1)
			k1 = 1;
		}
	fprintf(stderr, "%s ", what[k]);
	if (maxrownamelen) {
		strcpy(stub_end, ".row");
		if (f = fopen(filename, "r")) {
			for(j = 0; j <= i; j++)
				if (!next_line)
					break;
			fclose(f);
			if (j > i) {
				for(b = buf; *b; b++)
					if (*b == '\n') {
						*b = 0;
						break;
						}
				fprintf(stderr, "%s: ", buf);
				goto ret;
				}
			}
		}
	fprintf(stderr, nfmt[k1], i + 1);
 ret:
	fflush(stderr);
	}
