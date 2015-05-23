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

#define Egulp 400
#define GAP_MAX 10
/* #define FUNNEL_MIN 5 */
int FUNNEL_MIN = 5;
int vrefGulp = 100;
int maxfwd = 5, want_derivs = 1;
char *progname;
static int Done;

#ifdef __cplusplus
extern "C" {
#endif
#ifdef KR_headers
#define Void /* */
 extern void Mach();
 extern int ascanf();
 extern int bscanf();
#ifndef Just_Linear
 extern void funcadd();
 extern ufunc *dynlink();
 extern void show_funcs();
#endif /* Just_Linear */
#else
#define Void void
 extern void Mach(void);
 extern int ascanf(FILE*, const char*, ...);
 extern int bscanf(FILE*, const char*, ...);
#ifndef Just_Linear
 extern void funcadd(void);
 extern ufunc *dynlink(const char *);
 extern void show_funcs(void);
 static int compar(const void *a, const void *b);
#endif /* Just_Linear */
#endif

 edag_info edaginfo;
 int need_nl;
 static int k_seen, nv0;
 static int need_funcadd = 1;

#ifdef Just_Linear
#ifdef KR_headers
 real f_OPNUM(e) expr *e; {return 0.;}
#else
 real f_OPNUM(expr *e){return 0.;}
#endif

 static void
sorry_nonlin(Void)
{
	fprintf(stderr,
		"Sorry, %s can't handle nonlinearities.\n",
		progname ? progname : "");
	exit(1);
	}
#else /* Just_Linear */

#define NFHASH 23

 static func_info *fhash[NFHASH], **funcs, *funcsfirst, *funcslast;

#include "r_opn.hd"
 static int firstc1, lastc1, ncom_togo, nvref;
 static int *vrefnext, *vrefx;

 static derp *last_d;

 extern efunc *r_ops[];
 static relo *relolist, *relo2list;
 static expr_if *iflist, *if2list, *if2list_end;
 static expr_va *varglist, *varg2list, *varg2list_end;
 real edagread_one = 1.;
 static int nderp;
#undef nzc
 static int amax1, imap_len, last_cex, lasta, lasta0, lasta00, lastj,
	max_var, nocopy, nv01, nv011, nv0b, nv0c, nv1, nzc, nzclim;
 static int co_first = 1;
 static int *imap, *zc, *zci;
#ifdef KR_headers
 static expr *(*holread)();
#else
 static expr *(*holread)(FILE *);
#endif
#endif /* Just_Linear */

#ifdef KR_headers
 static int (*xscanf)();
#else
 static int (*xscanf)(FILE*, const char*, ...);
#endif

 int lineinc = 1;
 static int can_end;
 long Line;
 static char rl_buf[80];

 void
what_prog(Void)
{
	if (progname)
		fprintf(stderr, "%s: ", progname);
	}

 void
badread(Void)
{
	what_prog();
	fprintf(stderr, "error reading line %ld of %s:\n\t", Line, filename);
	}

 static int
#ifdef KR_headers
peek(fd) FILE* fd;
#else
peek(FILE* fd)
#endif
{
	int c;
	Line++;
	lineinc = 0;
	rl_buf[0] = c = getc(fd);
	return c;
	}

 void
#ifdef KR_headers
scream(fmt, j, k) char *fmt;
#else
scream(char *fmt, int j, int k)
#endif
{
	fprintf(stderr, fmt, j, k);
	exit(1);
	}

 static void
#ifdef KR_headers
memfailure(who, what, Len) char *who, *what; fint Len;
#else
memfailure(char *who, char *what, fint Len)
#endif
{
	fprintf(stderr, "%s(%lu) failure: %s.\n", who, (long)Len, what);
	exit(1);
	}

static char	too_large[] =	"problem too large",
		ran_out[] =	"ran out of memory";

#ifdef KR_headers
 char *
mymalloc(Len) fint Len;
#else
 void *
mymalloc(fint Len)
#endif
{
#ifdef KR_headers
	char *rv;
#else
	void *rv;
#endif
	static char who[] = "malloc";
	size_t len = (size_t) Len;
	if (sizeof(Len) != sizeof(len) && Len != (fint)len)
		memfailure(who, too_large, Len);
	rv = malloc(len);
	if (!rv)
		memfailure(who, ran_out, Len);
	return rv;
	}

#ifdef KR_headers
 char *
myralloc(rv, Len) char *rv; fint Len;
#else
 void *
myralloc(void *rv, fint Len)
#endif
{
	static char who[] = "realloc";
	size_t len = (size_t) Len;
	if (sizeof(Len) != sizeof(len) && Len != (fint)len)
		memfailure(who, too_large, Len);
	rv = realloc((char *)rv, len);
	if (!rv)
		memfailure(who, ran_out, Len);
	return rv;
	}

 static char *memLast, *memNext;

#ifdef EDAG_RESET
#define Mb_gulp 31
 typedef struct Mblock {
	struct Mblock *next;
	Char *m[Mb_gulp];
	} Mblock;

 static Mblock *Mb;
 static Char **Mbnext, **Mblast;

 static Char *
#ifdef KR_headers
Malloc1(n)
	fint n;
#else
Malloc1(fint n)
#endif
{
	Mblock *mb;

	if (Mbnext >= Mblast) {
		mb = (Mblock *)Malloc(sizeof(Mblock));
		mb->next = Mb;
		Mb = mb;
		Mbnext = mb->m;
		Mblast = mb->m + Mb_gulp;
		}
	return *Mbnext++ = Malloc(n);
	}

 void
#ifdef KR_headers
edag_reset(clear_edaginfo) int clear_edaginfo;
#else
edag_reset(int clear_edaginfo)
#endif
{
	Char **x, **x0;
	Mblock *mb;

	if (clear_edaginfo) {
		if (filename)
			free(filename);
		memset((char *)&edaginfo, 0, sizeof(edag_info));
		}
#ifndef Just_Linear
	if (!need_funcadd) {
		memset((char *)fhash, 0, NFHASH*sizeof(func_info *));
		need_funcadd = 1;
		funcs = 0;
		funcsfirst = 0;
		funcslast = 0;
		}
#endif
	if (!Done)
		return;
	Done = 0;
	k_seen = 0;
	can_end = 0;
#ifndef Just_Linear
	firstc1 = lastc1 = nvref = 0;
	vrefnext = vrefx = 0;
	relolist = relo2list = 0;
	last_d = 0;
	iflist = if2list = if2list_end = 0;
	varglist = varg2list = varg2list_end = 0;
	imap = 0;
	amax1 = imap_len = last_cex = lastj = nocopy = 0;
	zerograds = 0;
	co_first = 1;
#endif
	if (!Mb)
		return;
	x = Mbnext;
	Mbnext = Mblast = 0;
	memNext = memLast = 0;
	x0 = Mb->m;
	for(;;) {
		while(x > x0)
			free(*--x);
		mb = Mb->next;
		free(Mb);
		if (!(Mb = mb))
			break;
		x0 = Mb->m;
		x = x0 + Mb_gulp;
		}
	}
#else /* EDAG_RESET */
#define Malloc1 Malloc
#endif /* EDAG_RESET */

 void
badline(Void)
{
	fprintf(stderr, "bad line %ld of %s", Line, filename);
	if (xscanf == ascanf) {
		if (!lineinc)
			rl_buf[1] = 0;
		fprintf(stderr, ": %s\n", rl_buf);
		}
	else
		fprintf(stderr, "\n");
	exit(1);
	}

 char *
#ifdef KR_headers
read_line(nl) FILE *nl;
#else
read_line(FILE *nl)
#endif
{
	register char *s, *se;
	register int x;
	char *rv;

	rv = s = rl_buf + 1 - lineinc;
	se = rl_buf + sizeof(rl_buf) - 1;
	Line += lineinc;
	lineinc = 1;
	for(;;) {
		x = getc(nl);
		if (x < 0)
			break;
		if (x == '\n') {
			*s = 0;
			return rv;
			}
		*s++ = x;
		if (s >= se) {
			while((x = getc(nl)) != '\n' && x >= 0);
			break;
			}
		}
	if (x < 0) {
		if (can_end)
			return 0;
		fprintf(stderr,
			"edagread: premature end of file, line %ld of %s\n",
			Line, filename);
		exit(1);
		}
	*s = 0;
	return rv;
	}

#ifdef Double_Align
#define memadj(x) x
#else
#define memadj(x) (((x) + (sizeof(long)-1)) & ~(sizeof(long)-1))
#endif

 static char *
#ifdef KR_headers
mem(len) unsigned len;
#else
mem(unsigned len)
#endif
{
	char *rv;
	fint k;

#ifdef Double_Align
	len = (len + (sizeof(real)-1)) & ~(sizeof(real)-1);
#endif

	if (memNext + len >= memLast) {
		memNext = (char *)Malloc1(k = Egulp*Sizeof(expr) + len);
		memLast = memNext + k;
		}
	rv = memNext;
	memNext += len;
	return rv;
	}

#ifndef Just_Linear

 static void
#ifdef KR_headers
fscream(name, nargs, kind) char *name; char *kind;
#else
fscream(const char *name, int nargs, char *kind)
#endif
{
	fprintf(stderr, "line %ld: attempt to call %s with %d %sargs\n",
		Line, name, nargs, kind);
	exit(1);
	}

 static func_info *
#ifdef KR_headers
lookup(s, add) register char *s;
#else
lookup(register const char *s, int add)
#endif
{
	register unsigned x = 0;
	func_info *fi, **finext;
	Const char *s0 = s;

	while(*s)
		x = 31*x + *s++;
	finext = &fhash[x % NFHASH];
	for(fi = *finext; fi; fi = fi->next)
		if (!strcmp(s0, fi->name)) {
			if (add) {
				fprintf(stderr,
				"addfunc: duplicate function %s\n", s0);
				exit(1);
				}
			return fi;
			}
	if (add) {
		fi = (func_info *)mem(sizeof(func_info));
		fi->next = *finext;
		*finext = fi;
		fi->name = s0;
		}
	return fi;
	}

 void
#ifdef KR_headers
addfunc(fname, f, ftype, nargs) char *fname; ufunc *f;
#else
addfunc(const char *fname, ufunc *f, int ftype, int nargs)
#endif
{
	register func_info *fi;
	if (ftype && ftype != 1) {
		fprintf(stderr, "function %s: ", fname);
		scream("ftype = %d; expected 0 or 1\n", ftype, 0);
		}
	fi = lookup(fname, 1);
	fi->funcp = f;
	fi->ftype = ftype;
	fi->nargs = nargs;
	if (!funcsfirst)
		funcsfirst = fi;
	else
		funcslast->fnext = fi;
	funcslast = fi;
	fi->fnext = 0;
	}

 static void
func_add(Void)
{
	if (need_funcadd) {
		funcadd();
		need_funcadd = 0;
		}
	}

 void
show_funcs(Void)
{
	func_info *fi;
	int nargs;
	char *atleast;

	func_add();
	fprintf(stderr, "Available nonstandard functions:%s\n",
		(fi = funcsfirst) ? "" : " none");
	for(; fi; fi = fi->fnext) {
		if ((nargs = fi->nargs) >= 0)
			atleast = "";
		else {
			nargs = -(1 + nargs);
			atleast = "at least ";
			}
		fprintf(stderr, "\t%s(%s%d %sarg%s)\n", fi->name,
			atleast, nargs, fi->ftype ? "" : "real ",
			nargs == 1 ? "" : "s");
		}
	fflush(stderr);
	}

 static void
#ifdef KR_headers
new_derp(a, b, c) real *c;
#else
new_derp(int a, int b, real *c)
#endif
{
	derp *d;
	if (a == nv1)
		return;
	nderp++;
	d = (derp *)mem(sizeof(derp));
	d->next = last_d;
	last_d = d;
	d->a.i = a;
	d->b.i = b;
	d->c.rp = c;
	}

 static derp *
#ifdef KR_headers
new_relo(e, Dnext, ap) expr *e; register derp *Dnext; int *ap;
#else
new_relo(expr *e, register derp *Dnext, int *ap)
#endif
{
	register relo *r;
	register derp *d;

	r = (relo *)mem(sizeof(relo));
	r->next = relolist;
	r->next2 = relo2list;
	relo2list = relolist = r;
	if (last_d != Dnext) {
		*ap = e->a;
		for(d = last_d; d->next != Dnext; d = d->next);
		d->next = 0;
		}
	else {
		last_d = 0;
		new_derp(e->a, *ap = lasta++, &edagread_one);
		}
	r->D = r->Dcond = last_d;
	r->Dnext = Dnext;
	return r->D;
	}

 static relo *
#ifdef KR_headers
new_relo1(Dnext) derp *Dnext;
#else
new_relo1(derp *Dnext)
#endif
{
	register relo *r;

	r = (relo *)mem(sizeof(relo));
	r->next = relolist;
	relolist = r;
	r->D = 0;
	r->Dnext = Dnext;
	return r;
	}

 static expr *
#ifdef KR_headers
new_expr(opcode, L, R, deriv) expr *L, *R;
#else
new_expr(int opcode, expr *L, expr *R, int deriv)
#endif
{
	register expr *rv;
	efunc *o;
	real t;
	int L1, R1;

	o = r_ops[opcode];
	if (o == f_OPPOW)
		if (R->op == f_OPNUM)
			if ((t = ((expr_n *)R)->v) == 2.) {
				o = f_OP2POW;
				R = 0;
				}
			else
				o = f_OP1POW;
		else if (L->op == f_OPNUM)
			o = f_OPCPOW;
	rv = (expr *)mem(sizeof(expr));
	rv->op = o;
	rv->L.e = L;
	rv->R.e = R;
	rv->a = nv1;
	if (deriv) {
		L1 = L && L->op != f_OPNUM && L->a != nv1;
		R1 = R && R->op != f_OPNUM && R->a != nv1;
		if (L1 | R1) {
			rv->a = lasta++;
			if (L1)
				new_derp(L->a, rv->a, &rv->dL);
			if (R1)
				new_derp(R->a, rv->a, &rv->dR);
			}
		}
	return rv;
	}

 static char op_type[] = {
#include "op_type.hd"
	};

#endif /* Just_Linear */

 static expr *
#ifdef KR_headers
eread(nl, deriv) FILE *nl;
#else
eread(FILE *nl, int deriv)
#endif
{
	expr_n *rvn;
	short sh;
	fint L1;
	real r;
#ifndef Just_Linear
	char **sa;
	int a0, a1, *at, i, j, k, kd, ks, numargs, symargs;
	real *b, *ra;
	expr *L, *arg, **args, **argse, *rv;
	expr_va *rva;
	plterm *p;
	de *d;
	derp *dsave;
	efunc *op;
	expr_if *rvif;
	expr_f *rvf;
	func_info *fi;
	arglist *al;
	argpair *ap, *sap;
	static real dvalue[] = {
#include "dvalue.hd"
		};
#endif /* Just_Linear */

	switch(peek(nl)) {
#ifdef Just_Linear
		case 'f':
		case 'h':
		case 'o':
		case 'v':
			sorry_nonlin();
#else
		case 'f':
			if (xscanf(nl, "%d %d", &i, &j) != 2)
				badline();
			fi = funcs[i];
			if (fi->nargs >= 0) {
				if (fi->nargs != j) {
 bad_nargs:
					fscream(fi->name, j, "");
					}
				}
			else if (-(1+fi->nargs) > j)
				goto bad_nargs;
			rvf = (expr_f *)mem(sizeof(expr_f)
					+ (j-1)*sizeof(expr *));
			rvf->op = f_OPFUNCALL;
			rvf->f = fi->funcp;
			args = rvf->args;
			argse = args + j;
			k = ks = symargs = numargs = 0;
			while(args < argse) {
				arg = *args++ = eread(nl, deriv);
				if ((op = arg->op) == f_OPHOL)
					symargs++;
				else if (op == f_OPIFSYM)
					ks++;
				else {
					numargs++;
					if (op != f_OPNUM)
						k++;
					}
				}
			symargs += ks;
			if (symargs && !(fi->ftype & 1))
				fscream(fi->name, symargs, "symbolic ");
			kd = deriv && k ? numargs : 0;
			rvf->a = kd ? lasta++ : nv1;
			ra = (real *)mem(sizeof(arglist)
					+ (k+ks)*sizeof(argpair)
					+ (numargs+kd)*sizeof(real)
					+ symargs*sizeof(char *)
					+ j*sizeof(int));
			b = kd ? ra + numargs : ra;
			al = rvf->al = (arglist *)(b + numargs);
			al->n = numargs + symargs;
			al->ra = ra;
			al->derivs = kd ? (void *)b : 0;
			sa = al->sa = (char **)(al + 1);
			ap = rvf->ap = (argpair *)(sa + symargs);
			sap = rvf->sap = ap + k;
			at = al->at = (int *)(sap + ks);
			symargs = numargs = 0;
			for(args = rvf->args; args < argse; at++) {
				arg = *args++;
				if ((op = arg->op) == f_OPHOL) {
					*at = --symargs;
					*sa++ = ((expr_h *)arg)->sym;
					}
				else if (op == f_OPIFSYM) {
					*at = --symargs;
					sap->e = arg;
					(sap++)->u.s = sa++;
					}
				else {
					*at = numargs++;
					if (op == f_OPNUM)
						*ra = ((expr_n *)arg)->v;
					else  {
						ap->e = arg;
						(ap++)->u.v = ra;
						if (kd) {
							new_derp(arg->a,
								rvf->a, b);
							*b = 0;
							}
						}
					b++;
					ra++;
					}
				}
			rvf->ape = ap;
			rvf->sape = sap;
			return (expr *)rvf;

		case 'h':
			return holread(nl);
#endif /* Just_Linear */
		case 's':
			if (xscanf(nl, "%hd", &sh) != 1)
				badline();
			r = sh;
			goto have_r;

		case 'l':
			if (xscanf(nl, "%ld", &L1) != 1)
				badline();
			r = L1;
			goto have_r;

		case 'n':
			if (xscanf(nl, "%lf", &r) != 1)
				badline();
 have_r:
			rvn = (expr_n *)mem(size_expr_n);
			rvn->op = f_OPNUM;
			rvn->v = r;
			return (expr *)rvn;
#ifndef Just_Linear

		case 'o':
			break;

		case 'v':
			if (xscanf(nl,"%d",&k) != 1 || k < 0 || k > max_var)
				badline();
			if (k < nv01 && deriv && !zc[k]++)
				zci[nzc++] = k;
			return (expr *)(var_e + k);

#endif /* Just_Linear */
		default:
			badline();
		}

#ifndef Just_Linear

	if (xscanf(nl, "%d", &k) != 1 || k < 0 || k >= sizeof(op_type))
		badline();
	switch(op_type[k]) {

		case 1:	/* unary */
			rv = new_expr(k, eread(nl, deriv), 0, deriv);
			rv->dL = dvalue[k];	/* for UMINUS, FLOOR, CEIL */
			return rv;

		case 2:	/* binary */
			if (dvalue[k] == 11)
				deriv = 0;
			L = eread(nl, deriv);
			rv = new_expr(k, L, eread(nl, deriv), deriv);
			rv->dL = 1.;
			rv->dR = dvalue[k];	/* for PLUS, MINUS, REM */
			return rv;

		case 3:	/* vararg (min, max) */
			i = -1;
			xscanf(nl, "%d", &i);
			if (i <= 0)
				badline();
			rva = (expr_va *)mem(sizeof(expr_va));
			rva->op = r_ops[k];
			rva->L.d = d = (de *)mem(i*sizeof(de) + sizeof(expr *));
			rva->next = varglist;
			varglist = varg2list = rva;
			if (!last_d) {
				new_derp(lasta, lasta, &edagread_one);
				lasta++;
				}
			rva->d0 = dsave = last_d;
			a0 = a1 = lasta;
			for(j = 0; i > 0; i--, d++) {
				last_d = dsave;
				d->e = L = eread(nl, deriv);
				if (L->op == f_OPNUM || L->a == nv1) {
					d->d = dsave;
					d->dv.i = nv1;
					}
				else if (deriv) {
					d->d = new_relo(L, dsave, &d->dv.i);
					j++;
					if (a1 < lasta)
						a1 = lasta;
					lasta = a0;
					}
				}
			d->e = 0;	/* sentinnel expr * */
			last_d = dsave;
			if (j) {
				rva->a = lasta = a1;
				new_derp(0, lasta++, &edagread_one);
				/* f_MINLIST or f_MAXLIST will replace the 0 */
				rva->R.D = last_d;
				nocopy = 1;
				}
			else {
				rva->a = nv1;
				rva->R.D = 0;
				}
			return (expr *)rva;

		case 4: /* piece-wise linear */
			i = -1;
			xscanf(nl, "%d", &i);
			if (i <= 1)
				badline();
			plterms++;
			j = 2*i - 1;
			p = (plterm *)mem(sizeof(plterm) + (j-1)*sizeof(real));
			p->n = i;
			b = p->bs;
			do {
				switch(peek(nl)) {
					case 's':
						if (xscanf(nl,"%hd",&sh) != 1)
							badline();
						r = sh;
						break;
					case 'l':
						if (xscanf(nl,"%ld",&L1) != 1)
							badline();
						r = L1;
						break;
					case 'n':
						if (xscanf(nl,"%lf",&r) == 1)
							break;
					default:
						badline();
					}
				*b++ = r;
				}
				while(--j > 0);
			rv = (expr *)mem(sizeof(expr));
			rv->op = f_OPPLTERM;
			rv->L.p = p;
			rv->R.e = L = eread(nl, deriv);
			if (deriv)
				new_derp(L->a, rv->a = lasta++, &rv->dL);
			return rv;

		case 5: /* if */
			rvif = (expr_if *)mem(sizeof(expr_if));
			rvif->op = r_ops[k];
			rvif->next = iflist;
			iflist = if2list = rvif;
			if (!last_d) {
				new_derp(lasta, lasta, &edagread_one);
				lasta++;
				}
			rvif->d0 = dsave = last_d;
			rvif->e = eread(nl, 0);
			a0 = lasta;
			rvif->T = L = eread(nl, deriv);
			j = 0;
			if (L->op == f_OPNUM || L->a == nv1) {
				rvif->dT = dsave;
				rvif->Tv.i = nv1;
				}
			else if (j = deriv)
				rvif->dT = new_relo(L, dsave, &rvif->Tv.i);
			a1 = lasta;
			lasta = a0;
			last_d = dsave;
			rvif->F = L = eread(nl, deriv);
			if (L->op == f_OPNUM || L->a == nv1) {
				rvif->dF = dsave;
				rvif->Fv.i = nv1;
				}
			else if (j = deriv)
				rvif->dF = new_relo(L, dsave, &rvif->Fv.i);
			if (lasta < a1)
				lasta = a1;
			last_d = dsave;
			if (j) {
				new_derp(0, rvif->a = lasta++, &edagread_one);
				rvif->D = last_d;
				nocopy = 1;
				}
			else {
				rvif->a = nv1;
				rvif->D = 0;
				}
			return (expr *)rvif;

		case 6: /* sumlist */
			i = j = 0;
			xscanf(nl, "%d", &i);
			if (i <= 2)
				badline();
			rv = (expr *)mem(sizeof(expr) - sizeof(real)
					+ i*sizeof(expr *));
			rv->op = r_ops[k];
			a0 = lasta;
			rv->L.ep = args = (expr **)&rv->dR;
			if (deriv) {
				rv->a = lasta++;
				do {
					*args++ = L = eread(nl, deriv);
					if (L->op != f_OPNUM && L->a != nv1) {
						new_derp(L->a, rv->a,
							&edagread_one);
						j++;
						}
					}
					while(--i > 0);
				}
			else do
				*args++ = eread(nl, deriv);
				while(--i > 0);
			rv->R.ep = args;
			if (!j) {
				rv->a = nv1;
				lasta = a0;
				}
			return rv;
			}
#endif /* Just_Linear */
	badline();
	return 0;
	}

#ifndef Just_Linear

 static list *
#ifdef KR_headers
new_list(nxt) list *nxt;
#else
new_list(list *nxt)
#endif
{
	list *rv = (list *)mem(sizeof(list));
	rv->next = nxt;
	return rv;
	}

 static list *
crefs(Void)
{
	int i;
	list *rv = 0;

	while(nzc > 0) {
		if ((i = zci[--nzc]) >= nv0) {
			rv = new_list(rv);
			rv->item.i = i;
			}
		zc[i] = 0;
		}
	return rv;
	}

 static funnel *
#ifdef KR_headers
funnelfix(f) funnel *f;
#else
funnelfix(funnel *f)
#endif
{
	cexp *ce;
	funnel *fnext, *fprev;

	for(fprev = 0; f; f = fnext) {
		fnext = f->next;
		f->next = fprev;
		fprev = f;
		ce = f->ce;
		ce->z.i = ce->d->b.i;
		}
	return fprev;
	}

 static derp *
#ifdef KR_headers
derpadjust(d0, a, dnext) derp *d0; register int a; derp *dnext;
#else
derpadjust(derp *d0, register int a, derp *dnext)
#endif
{
	register derp *d, *d1;
	register int *r, *re;
	relo *rl;
	expr_if *il, *ile;
	expr_va *vl, *vle;
	de *de1;

	if (!(d = d0))
		return dnext;
	r = imap + lasta0;
	re = imap + lasta;
	while(r < re)
		*r++ = a++;
	if (amax < a)
		amax = a;
	r = imap;
	for(;; d = d1) {
		d->a.i = r[d->a.i];
		d->b.i = r[d->b.i];
		if (!(d1 = d->next))
			break;
		}
	d->next = dnext;
	if (rl = relo2list) {
		relo2list = 0;
		do {
			d = rl->Dcond;
			do {
				d->a.i = r[d->a.i];
				d->b.i = r[d->b.i];
				}
				while(d = d->next);
			}
			while(rl = rl->next2);
		}
	if (if2list != if2list_end) {
		ile = if2list_end;
		if2list_end = il = if2list;
		do {
			il->Tv.i = r[il->Tv.i];
			il->Fv.i = r[il->Fv.i];
			}
			while((il = il->next) != ile);
		}
	if (varg2list != varg2list_end) {
		vle = varg2list_end;
		varg2list_end = vl = varg2list;
		do {
			for(de1 = vl->L.d; de1->e; de1++)
				de1->dv.i = r[de1->dv.i];
			}
			while((vl = vl->next) != vle);
		}
	return d0;
	}

 static derp *
#ifdef KR_headers
derpcopy(ce, dnext) cexp *ce; derp *dnext;
#else
derpcopy(cexp *ce, derp *dnext)
#endif
{
	register derp	*d, *dprev;
	register int	*map;
	derp		d00;

	if (!(d = ce->d))
		return dnext;
	map = imap;
	for(dprev = &d00; d; d = d->next) {
		new_derp(map[d->a.i], map[d->b.i], d->c.rp);
		dprev = dprev->next = last_d;
		}
	dprev->next = dnext;
	return d00.next;
	}

 static void
imap_alloc(Void)
{
	int i, *r, *re;

	if (imap) {
		imap_len += lasta;
		imap = (int *)Realloc(imap, imap_len*Sizeof(int));
		return;
		}
	imap_len = amax1 > lasta ? amax1 : lasta;
	imap_len += 100;
	r = imap = (int *)Malloc(imap_len*Sizeof(int));
	for(i = 0, re = r + nv1+1; r < re;)
		*r++ = i++;
	}

 static int
#ifdef KR_headers
compar(a, b) char *a, *b;
#else
compar(const void *a, const void *b)
#endif
{ return *(int*)a - *(int*)b; }

 static void
#ifdef KR_headers
comsubs(alen, d, z) cde *d; int alen, **z;
#else
comsubs(int alen, cde *d, int **z)
#endif
{
	list *L;
	int a, i, j, k;
	int *r, *re, *z1;
	cexp *ce;
	derp *D, *dnext;
	relo *R;

	D = last_d;
	a = lasta00;
	dnext = 0;
	R = 0;
	for(i = j = 0; i < nzc; i++)
		if ((k = zci[i]) >= nv0)
			zci[j++] = k;
		else
			zc[k] = 0;
	if (nzc = j) {
		for(i = 0; i < nzc; i++)
			for(L = cexps[zci[i]-nv0].cref; L; L = L->next)
				if (!zc[L->item.i]++)
					zci[nzc++] = L->item.i;
		if (nzc > 1)
			if (nzc < nzclim)
				qsort((char *)zci, nzc, sizeof(int), compar);
			else for(i = nv0, j = 0; i < max_var; i++)
				if (zc[i])
					zci[j++] = i;
		}
	if (z && (k = lastc1 - firstc1 + nzc)) {
		i = (2*k + 1)*sizeof(int);
		*z = z1 = k > 20 ? (int *)Malloc1(i) : (int *)mem(i);
		*z1++ = k;
		}
	if (nzc > 0) {
		R = new_relo1(dnext);
		i = 0;
		do {
			j = zci[i];
			zc[j] = 0;
			ce = &cexps[j - nv0];
			if (ce->funneled)
				imap[var_e[j].a] = a++;
			else {
				r = imap + ce->z.i;
				re = r + ce->zlen;
				while(r < re)
					*r++ = a++;
				}
			if (z) {
				*z1++ = j;
				*z1++ = a - 1;
				}
			dnext = R->D = derpcopy(ce, R->D);
			}
			while(++i < nzc);
		nzc = 0;
		}
	if (D || R) {
		if (!R)
			R = new_relo1(dnext);
		D = R->D = derpadjust(D, a, R->D);
		if (d->e->op != f_OPVARVAL)
			d->e->a = imap[d->e->a];
		}
	if (z)
		for(i = firstc1 + nv01, j = lastc1 + nv01; i < j; i++) {
			*z1++ = i;
			*z1++ = imap[var_e[i].a];
			}
	d->d = D;
	a += alen;
	d->zaplen = (a > lasta00 ? a - nv1 : 0)*sizeof(real);
	if (amax < a)
		amax = a;
	}
#endif /* Just_Linear */

 static void
#ifdef KR_headers
co_read(nl, d, cexp1_end, k, z) FILE *nl; cde *d; int *cexp1_end, k, **z;
#else
co_read(FILE *nl, cde *d, int *cexp1_end, int k, int **z)
#endif
{
	d += k;
#ifndef Just_Linear
	lastc1 = last_cex - nv011;
	if (cexp1_end)
		cexp1_end[k+1] = lastc1;
	if (amax1 < lasta)
		amax1 = lasta;
	if (co_first) {
		co_first = 0;
		if (imap_len < lasta)
			imap_alloc();
		f_b = funnelfix(f_b);
		f_c = funnelfix(f_c);
		f_o = funnelfix(f_o);
		}
	if (!lastj) {
		lasta = lasta0;
		last_d = 0;
		}
	lastj = 0;
#endif /* Just_Linear */
	d->e = eread(nl, want_derivs);
#ifndef Just_Linear
	{	int alen;
		alen = lasta - lasta0;
		if (imap_len < lasta)
			imap_alloc();
		if (z) {
			z += k;
			*z = 0;
			}
		comsubs(alen, d, z);
		firstc1 = lastc1;
		}
#endif /* Just_Linear */
	}

#ifndef Just_Linear
 static linpart *
#ifdef KR_headers
linpt_read(nl, nlin) FILE *nl;
#else
linpt_read(FILE *nl, int nlin)
#endif
{
	linpart *L, *rv;

	if (nlin <= 0)
		return 0;
	L = rv = (linpart *)mem(nlin*sizeof(linpart));
	do {
		if (xscanf(nl, "%d %lf", &L->v.i, &L->fac) != 2)
			badline();
		L++;
		}
		while(--nlin > 0);
	return rv;
	}

 static int
#ifdef KR_headers
funnelkind(ce, ip) cexp *ce; int *ip;
#else
funnelkind(cexp *ce, int *ip)
#endif
{
	int i, j, k, nzc0, rv;
	int *vr, *vre;

	ce->vref = 0;
	if (!(nzc0 = nzc))
		return 0;
	for(i = k = rv = 0; i < nzc; i++)
		if ((j = zci[i]) < nv0) {
			if (k >= maxfwd)
				goto done;
			vrefx[k++] = j;
			}
		else  {
			if (!(vr = cexps[j-nv0].vref))
				goto done;
			vre = vr + *vr;
			while(++vr <= vre)
				if (!zc[*vr]++)
					zci[nzc++] = *vr;
			}
	if (k >= nvref) {
		nvref = (maxfwd + 1)*(ncom_togo < vrefGulp
					? ncom_togo : vrefGulp);
		vrefnext = (int *)Malloc1(nvref*Sizeof(int));
		}
	vr = ce->vref = vrefnext;
	*vr = k;
	vrefnext += i = k + 1;
	nvref -= i;
	for(i = 0; i < k; i++)
		*++vr = vrefx[i];
	if (nderp > 3*k && !nocopy) {
		*ip = k;
		return 2;
		}
	else {
 done:
		if (nocopy || nderp > 3*nzc0)
			rv = 1;
		}
	while(nzc > nzc0)
		zc[zci[--nzc]] = 0;
	return rv;
	}

 static void
#ifdef KR_headers
cexp_read(nl, k, nlin) FILE *nl;
#else
cexp_read(FILE *nl, int k, int nlin)
#endif
{
	int a, fk, i, j, la0, na;
	int *z1, **zp;
	funnel *f, **fp;
	register linpart *L, *Le;
	expr *e;
	cplist *cl, *cl0;
	cexp *ce;

	ce = cexps + k - nv0;
	L = ce->L = linpt_read(nl, ce->nlin = nlin);
	nocopy = 0;
	last_d = 0;
	ce->z.i = la0 = lasta;
	nderps += nderp;
	nderp = 0;
	e = ce->e = eread(nl, 1);
	if (la0 == lasta) {
		a = lasta++;
		if (e->op != f_OPNUM)
			new_derp(e->a, a, &edagread_one);
		}
	else
		a = e->a;
	var_e[k].a = a;
	ce->zlen = lasta - la0;
	for(Le = L + nlin; L < Le; L++) {
		new_derp(i = L->v.i, a, &L->fac);
		if (!zc[i]++)
			zci[nzc++] = i;
		}
	if (zp = zaC)
		*(zp += k - nv0) = 0;
	if (fk = funnelkind(ce, &i)) {
		/* arrange to funnel */
		fp = k < nv0b ? &f_b : k < nv0c ? &f_c : &f_o;
		ce->funneled = f = (funnel *)mem(sizeof(funnel));
		f->next = *fp;
		*fp = f;
		f->ce = ce;
		if (imap_len < lasta)
			imap_alloc();
		if (fk == 1) {
			f->fulld = last_d;
			a = lasta00;
			z1 = 0;
			if (zp) {
				for(i = j = 0; i < nzc; i++)
					if (zci[i] >= nv0)
						j++;
				if (j) {
					i = (2*j + 1)*sizeof(int);
					*zp = z1 = j > 20
						? (int *)Malloc1(i)
						: (int *)mem(i);
					*z1++ = j;
					}
				}
			for(i = nzc; --i >= 0; )
				if ((j = zci[i]) >= nv0) {
					if (z1) {
						*z1++ = j;
						*z1++ = a;
						}
					imap[var_e[j].a] = a++;
					}
			if ((na = ce->zlen) || a > lasta00)
				na += a - nv1;
			f->fcde.zaplen = na*sizeof(real);
			i = nzc;
			derpadjust(last_d, a, 0);
			}
		else {
			f->fulld = 0;
			f->fcde.e = e;
			comsubs(ce->zlen, &f->fcde, zp);
			memcpy((char *)zci, (char *)vrefx, i*sizeof(int));
			}
		last_d = 0;
		cl0 = 0;
		while(i > 0)
			if ((a = var_e[zci[--i]].a) != nv1) {
				new_derp(a, lasta0, 0);
				cl = (cplist *)mem(sizeof(cplist));
				cl->next = cl0;
				cl0 = cl;
				cl->ca.i = imap[last_d->a.i];
				cl->cfa = last_d->c.rp =
						(real *)mem(sizeof(real));
				}
		f->cl = cl0;
		var_e[k].a = lasta0++;
		lasta = lasta0;
		}
	lasta0 = lasta;
	ce->d = last_d;
	ce->cref = crefs();
	--ncom_togo;
	}

 static void
#ifdef KR_headers
cexp1_read(nl, j, k, nlin) FILE *nl;
#else
cexp1_read(FILE *nl, int j, int k, int nlin)
#endif
{
	register linpart *L, *Le;
	register cexp1 *ce = cexps1 + (k - nv01);
	expr *e;
	int la0;

	L = ce->L = linpt_read(nl, ce->nlin = nlin);

	if (!lastj) {
		last_d = 0;
		if (amax1 < lasta)
			amax1 = lasta;
		lasta = lasta0;
		lastj = j;
		}
	la0 = lasta;
	e = ce->e = eread(nl, 1);
	if (la0 == lasta) {
		j = lasta++;
		if (e->op != f_OPNUM)
			new_derp(e->a, j, &edagread_one);
		}
	else
		j = e->a;
	var_e[k].a = j;
	for(Le = L + nlin; L < Le; L++)
		new_derp(L->v.i, j, &L->fac);
	last_cex = k;
	}

 static void
#ifdef KR_headers
ifadjust(e) register expr_if *e;
#else
ifadjust(register expr_if *e)
#endif
{
	for(; e; e = e->next) {
		e->Tv.rp = &adjoints[e->Tv.i];
		e->Fv.rp = &adjoints[e->Fv.i];
		}
	}

 static void
#ifdef KR_headers
vargadjust(e) register expr_va *e;
#else
vargadjust(register expr_va *e)
#endif
{
	register de *d;

	for(; e; e = e->next) {
		for(d = e->L.d; d->e; d++)
			d->dv.rp = &adjoints[d->dv.i];
		}
	}

 static void
#ifdef KR_headers
funneladj1(f) register funnel *f;
#else
funneladj1(register funnel *f)
#endif
{
	register real	*a	= adjoints;
	register derp	*d;
	register cplist	*cl;

	for(a = adjoints; f; f = f->next) {
		if (d = f->fulld) {
			f->fcde.d = d;
			do {
				d->a.rp = &a[d->a.i];
				d->b.rp = &a[d->b.i];
				}
				while(d = d->next);
			}
		for(cl = f->cl; cl; cl = cl->next)
			cl->ca.rp = &a[cl->ca.i];
		}
	}

 static void
funneladjust(Void)
{
	register cexp *c, *ce;
	register linpart *L, *Le;
	c = cexps;
	for(ce = c + ncom0; c < ce; c++)
		if (L = c->L)
			for(Le = L + c->nlin; L < Le; L++)
				L->v.rp = &var_e[L->v.i].v;

	funneladj1(f_b);
	funneladj1(f_c);
	funneladj1(f_o);
	}

 static void
com1adjust(Void)
{
	register cexp1 *c, *ce;
	register linpart *L, *Le;

	for(c = cexps1, ce = c + ncom1; c < ce; c++)
		for(L = c->L, Le = L + c->nlin; L < Le; L++)
			L->v.rp = &var_e[L->v.i].v;
	}
#endif /* Just_Linear */

 static void
goff_comp(Void)
{
	register int *ka = A_colstarts + 1;
	cgrad **cgx, **cgxe;
	register cgrad *cg;

	cgx = Cgrad;
	cgxe = cgx + n_con;
	while(cgx < cgxe)
		for(cg = *cgx++; cg; cg = cg->next)
			cg->goff = ka[cg->varno]++;
	}

 static void
colstart_inc(Void)
{
	register int *ka, *kae;
	ka = A_colstarts;
	kae = ka + nv0;
	while(ka <= kae)
		++*ka++;
	}

 static void
zerograd_chk(Void)
{
	int j, n, *z, **zg;
	ograd *og, **ogp, **ogpe;

	ogp = Ograd;
	ogpe = ogp + n_obj;
	j = n_obj;
	while(ogp < ogpe) {
		og = *ogp++;
		n = 0;
		while(og) {
			j += og->varno - n;
			n = og->varno + 1;
			og = og->next;
			}
		j += nv0 - n;
		}
	if (j == n_obj)
		return;
	zerograds = zg = (int **)mem(n_obj*sizeof(int*)+j*sizeof(int));
	z = (int*)(zg + n_obj);
	ogp = Ograd;
	while(ogp < ogpe) {
		*zg++ = z;
		og = *ogp++;
		n = 0;
		while(og) {
			while(n < og->varno)
				*z++ = n++;
			og = og->next;
			n++;
			}
		while(n < nv0)
			*z++ = n++;
		*z++ = -1;
		}
	}

 static void
adjust(Void)
{
#ifndef Just_Linear
	register derp *d;
	register real *a = adjoints;
	register relo *r;

	for(r = relolist; r; r = r->next) {
		for(d = r->D;; d = d->next) {
			d->a.rp = &a[d->a.i];
			d->b.rp = &a[d->b.i];
			if (!d->next)
				break;
			}
		d->next = r->Dnext;
		}
	ifadjust(iflist);
	vargadjust(varglist);
	if (ncom0)
		funneladjust();
	com1adjust();
	if (n_obj)
		zerograd_chk();
#endif /* Just_Linear */
	if (k_seen) {
		if (!A_vals)
			goff_comp();
		else if (Fortran)
			colstart_inc();
		}
	}

 static void
#ifdef KR_headers
br_read(nl, nc, Lp, U) FILE *nl; real **Lp, *U;
#else
br_read(FILE *nl, int nc, real **Lp, real *U)
#endif
{
	int i, inc;
	real *L;

	if (!(L = *Lp))
		L = *Lp = (real *)Malloc1(2*sizeof(real)*nc);
	if (U)
		inc = 1;
	else {
		U = L + 1;
		inc = 2;
		}
	xscanf(nl, ""); /* purge line */
	for(i = 0; i < nc; i++, L += inc, U += inc) {
		switch(peek(nl) - '0') {
		  case 0:
			if (xscanf(nl,"%lf %lf",L,U)!= 2)
				badline();
			break;
		  case 1:
			if (xscanf(nl, "%lf", U) != 1)
				badline();
			*L = negInfinity;
			break;
		  case 2:
			if (xscanf(nl, "%lf", L) != 1)
				badline();
			*U = Infinity;
			break;
		  case 3:
			*L = negInfinity;
			*U = Infinity;
			xscanf(nl, ""); /* purge line */
			break;
		  case 4:
			if (xscanf(nl, "%lf", L) != 1)
				badline();
			*U = *L;
			break;
		  default:
			badline();
		  }
		}
	}

#ifndef Just_Linear

 static expr *
#ifdef KR_headers
aholread(nl) FILE *nl;
#else
aholread(FILE *nl)
#endif
{
	int i, k;
	expr_h *rvh;
	char *s, *s1;

	s = read_line(nl);
	k = *s;
	if (k < '1' || k > '9')
		badline();
	i = k - '0';
	while((k = *++s) != ':') {
		if (k < '0' || k > '9')
			badline();
		i = 10*i + k - '0';
		}
	rvh = (expr_h *)mem(memadj(sizeof(expr_h) + i));
	for(s1 = rvh->sym; *s1 = *++s; s1++)
		if (--i < 0)
			badline();
	if (i) {
		for(*s1++ = '\n'; --i > 0; ) {
			k = getc(nl);
			if (k <= 0)
				badline();
			if ((*s1++ = k) == '\n')
				Line++;
			}
		if (getc(nl) != '\n')
			badline();
		}
	*s1 = 0;
	rvh->op = f_OPHOL;
	rvh->a = nv1;
	return (expr *)rvh;
	}

 static expr *
#ifdef KR_headers
bholread(nl) FILE *nl;
#else
bholread(FILE *nl)
#endif
{
	int i;
	expr_h *rvh;
	char *s;

	if (xscanf(nl, "%d", &i) != 1)
		badline();
	rvh = (expr_h *)mem(memadj(sizeof(expr_h) + i));
	s = rvh->sym;
	if (fread(s, i, 1, nl) != 1)
		badline();
	s[i] = 0;
	rvh->op = f_OPHOL;
	rvh->a = nv1;
	for(;;) switch(*s++) {
			case 0: goto break2; /* so we return at end of fcn */
			case '\n': Line++;
			}
 break2:
	return (expr *)rvh;
	}

 static void
#ifdef KR_headers
nlvzap(i, j)
	int i;
	int j;
#else
nlvzap(int i, int j)
#endif
{
	i -= j;
	while(--j >= 0)
		var_e[i+j].a = nv1;
	}
#endif /* Just_Linear */

 void
#ifdef KR_headers
edagread(nl) FILE *nl;
#else
edagread(FILE *nl)
#endif
{
	int i, i1, j, k, nc, nco, no, nv, nvc, nvo, nz;
	unsigned x;
	expr_v *e;
	cgrad *cg, *cg1;
	ograd *og, **ogp;
	real t;
	int *ka;
#ifndef Just_Linear
	int maxfwd1, ncom, nlin;
	func_info *fi;
	char fname[128];
#endif /* Just_Linear */

	if (Done) {
		fprintf(stderr,
		"edagread invoked twice without a preceding edag_reset().\n");
		exit(1);
		}
	Done = 1;
	Mach();
	if (!size_expr_n)	size_expr_n = sizeof(expr_n);
#ifdef Just_Linear
	/* Tried "xscanf = binary ? bscanf : ascanf;", but this
	 * encounters a bug in SGI's C compiler.
	 */
	if (binary_nl)
		xscanf = bscanf;
	else
		xscanf = ascanf;
#else
	if (c_cexp1st)
		*c_cexp1st = 0;
	if (o_cexp1st)
		*o_cexp1st = comc1;
	func_add();
	if (binary_nl) {
		holread = bholread;
		xscanf = bscanf;
		}
	else {
		holread = aholread;
		xscanf = ascanf;
		}

	ncom = comb + comc + como + comc1 + como1;
#endif /* Just_Linear */
	nc = n_con;
	no = n_obj;
	nvc = c_vars;
	nvo = o_vars;
	if (no < 0 || (nco = nc + no) <= 0)
		scream("edagread: nc = %d, no = %d\n", nc, no);
	if (pi0) {
		memset((char *)pi0, 0, nc*sizeof(real));
		if (havepi0)
			memset(havepi0, 0, nc);
		}
#ifdef Just_Linear
	nv = nv0 = nvc > nvo ? nvc : nvo;
	x = nco*sizeof(cde) + no*sizeof(ograd *) + nv*sizeof(expr_v) + no;
#else
	nv1 = nv0 = nvc > nvo ? nvc : nvo;
	max_var = nv = nv0 + ncom;
	combc = comb + comc;
	ncom0 = ncom_togo = combc + como;
	nzclim = ncom0 >> 3;
	ncom1 = comc1 + como1;
	nv0b = nv0 + comb;
	nv0c = nv0b + comc;
	nv01 = nv0 + ncom0;
	last_cex = nv011 = nv01 - 1;
	amax = lasta = lasta0 = lasta00 = nv1 + 1;
	if ((maxfwd1 = maxfwd + 1) > 1)
		nvref = maxfwd1*((ncom0 < vrefGulp ? ncom0 : vrefGulp) + 1);
	x = nco*sizeof(cde) + no*sizeof(ograd *)
		+ nv*(sizeof(expr_v) + 2*sizeof(int))
		+ ncom0*sizeof(cexp)
		+ ncom1*sizeof(cexp1)
		+ nfunc*sizeof(func_info *)
		+ nvref*sizeof(int)
		+ no;
#endif /* Just_Linear */
	if (X0)
		memset((char *)X0, 0, nv0*sizeof(real));
	if (havex0)
		memset(havex0, 0, nv0);
	e = var_e = (expr_v *)Malloc1(x);
	memset((char *)e, 0, (int)x);
	con_de = (cde *)(e + nv);
	obj_de = con_de + nc;
	Ograd = (ograd **)(obj_de + no);
	if (A_vals) {
		if (!A_rownos)
			A_rownos = (int *)Malloc1(edaginfo.nzc_*sizeof(int));
		}
	else if (nc)
		Cgrad = (cgrad **)Malloc1(nc*sizeof(cgrad *));
#ifdef Just_Linear
	objtype = (char *)(Ograd + no);
#else
	var_ex = e + nv0;
	var_ex1 = var_ex + ncom0;
	for(k = 0; k < nv; e++) {
		e->op = f_OPVARVAL;
		e->a = k++;
		}
#ifndef Just_Linear
	if (skip_int_derivs) {
		if (nlvbi)
			nlvzap(nlvb, nlvbi);
		if (nlvci)
			nlvzap(nlvb+nlvc, nlvci);
		if (nlvoi)
			nlvzap(nlvb+nlvc+nlvo, nlvoi);
		}
#endif
	cexps = (cexp *)(Ograd + no);
	cexps1 = (cexp1 *)(cexps + ncom0);
	funcs = (func_info **)(cexps1 + ncom1);
	zc = (int *)(funcs + nfunc);
	zci = zc + nv;
	vrefx = zci + nv;
	objtype = (char *)(vrefx + nvref);
	if (nvref) {
		vrefnext = vrefx + maxfwd1;
		nvref -= maxfwd1;
		}
	last_d = 0;
#endif
	ka = 0;
	nz = 0;
	for(;;) {
		can_end = 1;
		i = peek(nl);
		if (i == EOF) {
#ifndef Just_Linear
			free((char *)imap);
			/* Make amax long enough for nlc to handle */
			/* var_e[i].a for common variables i. */
			if (ncom0) {
				i = comb + como;
				if (i < combc)
					i = combc;
				if ((i += nv1 + 1) > amax)
					amax = i;
				}
			adjoints = (real *)Malloc1(amax*Sizeof(real));
			adjoints_nv1 = &adjoints[nv1];
			memset((char *)adjoints, 0, amax*sizeof(real));
			nderps += nderp;
#endif /* Just_Linear */
			adjust();
			nzjac = nz;
			if (!Lastx)
				Lastx = (real *)Malloc1(nv0*sizeof(real));
			fclose(nl);
			return;
			}
		can_end = 0;
		k = -1;
		switch(i) {
			case 'C':
				xscanf(nl, "%d", &k);
				if (k < 0 || k >= nc)
					badline();
				co_read(nl, con_de, c_cexp1st, k, zac);
				break;
#ifdef Just_Linear
			case 'F':
			case 'V':
				sorry_nonlin();
#else
			case 'F':
				if (xscanf(nl, "%d %d %d %127s",
						&i, &j, &k, fname) != 4
				|| i < 0 || i >= nfunc)
					badline();
				if (fi = lookup(fname,0)) {
					if (fi->nargs != k && fi->nargs >= 0
					 && (k >= 0 || fi->nargs < -(k+1))) {
						fprintf(stderr,
				"function %s: disagreement of nargs: ",
							fname);
						scream("%d and %d\n",
					 		fi->nargs, k);
						}
					}
				else {
					fi = (func_info *)mem(sizeof(func_info));
					fi->ftype = j;
					fi->nargs = k;
					fi->funcp = 0;
					fi->name = (Const char *)
					  strcpy(mem(memadj(strlen(fname)+1)),
							fname);
					}
				if (!fi->funcp && !(fi->funcp = dynlink(fname))){
					fprintf(stderr, "function %s ", fname);
					scream("not available\n",0,0);
					}
				funcs[i] = fi;
				break;
			case 'V':
				if (xscanf(nl, "%d %d %d", &k, &nlin, &j) != 3
				 || k < nv0 || k >= nv)
					badline();
				if (j)
					cexp1_read(nl, j, k, nlin);
				else
					cexp_read(nl, k, nlin);
				break;
#endif /* Just_Linear */
			case 'G':
				if (xscanf(nl, "%d %d", &j, &k) != 2
				|| j < 0 || j >= no || k <= 0 || k > nvo)
					badline();
				ogp = Ograd + j;
				while(k--) {
					*ogp = og = (ograd *)mem(sizeof(ograd));
					ogp = &og->next;
					if (xscanf(nl, "%d %lf", &og->varno,
							&og->coef) != 2)
						badline();
					}
				*ogp = 0;
				break;
			case 'J':
				if (xscanf(nl, "%d %d", &j, &k) != 2
				|| j < 0 || j >= nc || k <= 0 || k > nvc)
					badline();
				nz += k;
				if (ka) {
					if (!A_vals)
						goto cg_read;
					j += Fortran;
					while(k--) {
						if (xscanf(nl, "%d %lf",
							&i, &t) != 2)
							badline();
						i1 = ka[i]++;
						A_vals[i1] = t;
						A_rownos[i1] = j;
						}
					break;
					}
 cg_read:
				cg = 0;
				while(k--) {
					cg1 = (cgrad *)mem(sizeof(cgrad));
					cg1->next = cg;
					cg = cg1;
					if (ka) {
						if (xscanf(nl, "%d %lf", &cg->varno,
					    			&cg->coef) != 2)
							badline();
						}
					else
						if (xscanf(nl, "%d %d %lf", &cg->varno,
					    		    &cg->goff, &cg->coef) != 3)
							badline();
					}
				Cgrad[j] = cg;
				break;
			case 'O':
				if (xscanf(nl, "%d %d", &k, &j) != 2
				 || k < 0 || k >= no)
					badline();
				objtype[k] = j;
				co_read(nl, obj_de, o_cexp1st, k, zao);
				break;
			case 'r':
				br_read(nl, nc, &LUrhs, Urhsx);
				break;
			case 'b':
				br_read(nl, nv0, &LUv, Uvx);
				break;
			case 'k':
				k_seen++;
				k = nv0;
				if (!xscanf(nl,"%d",&j) || j != k - 1)
					badline();
				if (!(ka = A_colstarts))
					ka = A_colstarts = (int *)
						Malloc1((k+1)*Sizeof(int));
				*ka++ = 0;
				*ka++ = 0;	/* sic */
				while(--k > 0)
					if (!xscanf(nl, "%d", ka++))
						badline();
				ka = A_colstarts + 1;
				break;
			case 'x':
				if (!xscanf(nl,"%d",&k)
				|| k < 0 || k > nv0)
					badline();
				if (!X0 && want_xpi0 & 1) {
					X0 = (real *)Malloc1(nv0*sizeof(real));
					memset((char *)X0, 0, nv0*sizeof(real));
					}
				while(k--) {
					if (xscanf(nl, "%d %lf", &j, &t) != 2
					 || j < 0 || j >= nv)
						badline();
					if (X0) {
						X0[j] = t;
						if (havex0)
							havex0[j] = 1;
						}
					}
				break;
			case 'd':
				if (!xscanf(nl,"%d",&k)
				|| k < 0 || k > nc)
					badline();
				if (!pi0 && want_xpi0 & 2) {
					pi0 = (real *)Malloc1(nc*sizeof(real));
					memset((char *)pi0, 0, nc*sizeof(real));
					}
				while(k--) {
					if (xscanf(nl, "%d %lf", &j, &t) != 2
					 || j < 0 || j >= nc)
						badline();
					if (pi0) {
						pi0[j] = t;
						if (havepi0)
							havepi0[j] = 1;
						}
					}
				break;
			default:
				badline();
			}
		}
	}
#ifdef __cplusplus
	}
#endif
