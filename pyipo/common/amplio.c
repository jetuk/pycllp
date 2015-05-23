/****************************************************************
Copyright (C) Lucent Technologies 1996
All Rights Reserved

Permission to use, copy, modify, and distribute this software and
its documentation for any purpose and without fee is hereby
granted, provided that the above copyright notice appear in all
copies and that both that the copyright notice and this
permission notice and warranty disclaimer appear in supporting
documentation, and that the name of Lucent or any of its entities
not be used in advertising or publicity pertaining to
distribution of the software without specific, written prior
permission.

LUCENT DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL LUCENT OR ANY OF ITS ENTITIES BE LIABLE FOR ANY
SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
****************************************************************/

#include "jacdim.h"
#define MALLOC_defined
#include "lp.h"
#include "r_opn.hd"	/* for N_OPS */
#include "myalloc.h"

void printmat(
	int m,
	int n,
	int *iA,
	int *kA,
	double *A
);

void print_kp( LP *kp );

efunc *r_ops[N_OPS];

 extern char version[], *progname;
 static int n_badopts;

 struct options
{
	double iinftol;
	double inftol2;
	int maxflag;
        int itnlim;
        double timlim;
        double epssol;
        double epsnum;
        double epscdn;
        double stablty;
        int method;
        int dense;
        int pdf;
	int sf_req;
	/* int split; */
	int v;
	int objno;
	/* names for MPS format */
	char bounds[12];
	char obj[12];
	char ranges[12];
	char rhs[12];
	};

#ifdef MSDOS
 static struct options opns = {1e-5, 1, 0, 200, 10000000, 1e-6, 0, 1e-12, 1, 1,
 -1, 0, 8, 0, 1};
#else
 static struct options opns = {1e-5, 1, 0, 200, 10000000, 1e-6, 0, 1e-12, 1, 1,
 -1, 0, 8, 0, 1};
#endif

 static int show_version, time_flag;
 extern double xectim_();

#define known_kind	0
#define int_kind	1
#define dbl_kind	2
#define char_kind	3

 typedef struct option_word {
	char *name;
	int kind;
	int known_val;
	void *ival;
	char *label;
	} option_word;
 static option_word keywds[] = {
	{ "bounds",	char_kind,	0,	(void *)opns.bounds,
			"Bounds (BOUNDS)" },
        { "dense",      int_kind,       0,      (void *)&opns.dense,
                        "Dense column/row threshold (DENSE)" },
        { "dual",       known_kind,     2,      (void *)&opns.pdf,
                        "Oder to favor dual (DUAL)" },
        { "epscdn",     dbl_kind,       0,      (void *)&opns.epscdn,
                        "Epsilon for conditioning (EPSCDN)" },
        { "epsnum",     dbl_kind,       0,      (void *)&opns.epsnum,
                        "Epsilon in numerical factorization (EPSNUM)" },
        { "epssol",     dbl_kind,       0,      (void *)&opns.epssol,
                        "Epsilon in forward/backward solve (EPSSOL)" },
        { "iinftol",     dbl_kind,       0,      (void *)&opns.iinftol,
			"Feasibility tolerance (INFTOL)" },
        { "inftol2",    dbl_kind,       0,      (void *)&opns.inftol2,
                        "Feasibility for stopping rule (INFTOL2)" },
        { "iterlim",    int_kind,       0,      (void *)&opns.itnlim,
			"Iteration Limit (ITERLIM)" },
	{ "max",	known_kind,	-1,	(void *)&opns.maxflag,
			"Maximize (MAX)" },
	{ "maximize",	known_kind,	-1,	(void *)&opns.maxflag,
			"Maximize (MAX)" },
	{ "min",	known_kind,	1,	(void *)&opns.maxflag,
			"Minimize (MIN)" },
        { "mindeg",     known_kind,     1,      (void *)&opns.method,
                        "Minimum-degree reordering (MINDEG)" },
	{ "minimize",	known_kind,	1,	(void *)&opns.maxflag,
			"Minimize (MIN)" },
        { "minlocfil",  known_kind,     2,      (void *)&opns.method,
                        "Minimum-local-fill reordering (MINLOCFIL)" },
        { "noreord",    known_kind,     0,      (void *)&opns.method,
                        "No matrix reordering (NOREORD)" },
	{ "obj",	char_kind,	0,	(void *)opns.obj,
			"Objective vector (OBJ)" },
	{ "objno",	int_kind,	0,	(void *)&opns.objno,
			"Objective number (OBJNO; first = 0)" },
	{ "outlev",	int_kind,	0,	(void *)&opns.v,
			"Output Level (Outlev == VERBOSE)" },
        { "primal",     known_kind,     1,      (void *)&opns.pdf,
                        "Order to favor primal (PRIMAL)" },
	{ "ranges",	char_kind,	0,	(void *)opns.ranges,
			"Ranges (RANGES)" },
	{ "rhs",	char_kind,	0,	(void *)opns.rhs,
			"Right hand side (RHS)" },
	{ "sigfig",	int_kind,	0,	(void *)&opns.sf_req,
			"Significant digits (SIGFIG)" },
        { "stablty",    dbl_kind,       0,      (void *)&opns.stablty,
                        "Factor for stability (STABLTY)" },
	{ "timing",	int_kind,	0,	(void *)&time_flag,
		"Timing flag (TIMING: 1 = stdout, 2 = stderr, 3 = both)" },
        { "timlim",     dbl_kind,       0,      (void *)&opns.timlim,
                        "Time limit (in seconds) (TIMLIM)" },
	{ "verbose",	int_kind,	0,	(void *)&opns.v,
			"Verbosity level (VERBOSE)" },
	{ "version",	known_kind, 	1,	(void *)&show_version,
			version },
	};

#ifdef KR_headers
 extern char *getenv();
#define VOID
#define Void char
#else
#define VOID void
#define Void void
#endif

 char *
#ifdef KR_headers
get_option(s) char *s;
#else
get_option(char *s)
#endif
{
	char *s1;
	option_word *ow = keywds;

	if (ow = (option_word *)b_search((char *)keywds,
			(int)sizeof(option_word),
			(int)(sizeof(keywds)/sizeof(option_word)), &s)) {
		switch(ow->kind) {
			case known_kind:
				/* only accept first specification */
				if (!*(int *)ow->ival) {
					*(int *)ow->ival = ow->known_val;
					printf("%s\n", ow->label);
					need_nl = 0;
					}
				return s;
			case int_kind:
				*(int *)ow->ival = (int)strtol(s,&s1,10);
				printf("%s = %d\n", ow->label,
					*(int *)ow->ival);
				need_nl = 0;
				break;
			case dbl_kind:
				*(double *)ow->ival = (double)strtod(s,&s1);
				printf("%s = %g\n", ow->label,
					*(double *)ow->ival);
				need_nl = 0;
				break;
			case char_kind:
				strncpy((char *)ow->ival, s, 8);
				printf("%s = %s\n", ow->label,
					(char *)ow->ival);
				need_nl = 0;
				for(s1 = s, s += 8; *s1 && s1 < s; s1++);
			}
		}
	else {
		if (!*s)
			return s;
		for(s1 = s; *s1 > ' ' && *s1 != '='; s1++);
		printf("Unknown keyword \"%.*s\"\n", s1-s, s);
		if (*s1 == '=')
			while(*++s1 > ' ');
		n_badopts++;
		}
	return s1;
	}

 static void
#ifdef KR_headers
nonlin(n, what) char *what;
#else
nonlin(int n, char *what)
#endif
{
	if (n) {
		fprintf(stderr, "%s contains %s\n", filename, what);
		exit(4);
		}
	}

#undef INTEGER
#undef DOUBLE
#define INTEGER(x) (int *)Malloc((x)*Sizeof(int))
#define DOUBLE(x) (double *)Malloc((x)*Sizeof(double))

 int
#ifdef KR_headers
amplin(stub, kp) char *stub; LP *kp;
#else
amplin(char *stub, LP *kp)
#endif
{
	long M, N, NO, NZ, MXROW, MXCOL;
	int *ia, *ka, *kat, *varsgn;
	double *a, *a1, *ae, *b, *b1, *c, *l, *lu, *lue, *r, *r1, *u;
	cgrad *cg, **cgx;
	ograd *og;
	expr *e;
	int i, j;

	return_nofile = 1;
	for(i = 0; i < N_OPS; i++)
		r_ops[i] = (efunc *)i;
	want_derivs = 0;

	if (jacdim_(stub, &M, &N, &NO, &NZ, &MXROW, &MXCOL, (long)strlen(stub)))
		return 1;

	kp->c = c = DOUBLE(N);
	memset(c, 0, N*sizeof(double));
	kp->f = 0;

	qpcheck((long **)&kp->iQ, (long **)&kp->kQ, &kp->Q); 
	if (kp->kQ == NULL) {
	    kp->kQ = INTEGER(N+1);
	    kp->iQ = INTEGER(0);
	    kp->Q  = DOUBLE(0);
	    for (j=0; j<=N; j++) { kp->kQ[j] = 0; }
	}
	kp->qnz = kp->kQ[N];

	e = 0;
	if ((i = opns.objno - 1) >= 0 && i < n_obj) {
		obj_no = i;
		for(og = Ograd[i]; og; og = og->next)
			c[og->varno] = og->coef;
		if (!opns.maxflag) {
			opns.maxflag = objtype[i] ? -1 : 1;
		}
		if ((e = obj_de[i].e) && e->op == f_OPNUM) {
			kp->f = ((expr_n *)e)->v;
			e = 0;
			}
		}
	nonlin(nlc, "nonlinear constraints");
	nonlin(plterms, "piecewise-linear terms");

	kp->A = a = DOUBLE(NZ);
	kp->m = M;
	kp->n = N;
	kp->nz = NZ;
	kp->b = b = DOUBLE(M);
	kp->r = r = DOUBLE(M);
	kp->l = l = DOUBLE(N);
	kp->u = u = DOUBLE(N);
	kp->kA = ka = INTEGER(N+1);
	ka[N] = NZ;
	for(i = N; --i >= 0; )
		ka[i] = -1;
	kp->iA = ia = INTEGER(NZ);
	kp->varsgn = varsgn = INTEGER(N);
	kp->kAt = INTEGER(M+1);
	kp->iAt = INTEGER(NZ);
	kp->At = DOUBLE(NZ);
	kat = kp->kAt + M;
	j = *kat = NZ;
	i = (int)M;
	cgx = Cgrad + i;
	lu = LUrhs + 2*(i-1);
	b1 = b + i;
	r1 = r + i;
	while(--i >= 0) {
		--r1;
		--cgx;
		--b1;
		if (lu[1] >= Infinity) {
			*r1 = HUGE_VAL;
			*b1 = lu[0];
			}
		else if (lu[0] <= negInfinity) {
			*r1 = HUGE_VAL;
			*b1 = -lu[1];
			for(cg = *cgx; cg; cg = cg->next)
				cg->coef = -cg->coef;
			}
		else {
			*r1 = lu[1] - lu[0];
			*b1 = lu[0];
			}
		lu -= 2;
		for(cg = *cgx; cg; --j, cg = cg->next) {
			ia[ka[cg->varno] = cg->goff] = i;
			a[cg->goff] = cg->coef;
			}
		*--kat = j;
		}
	for(i = N; --i >= 0; )
		if (ka[i] == -1)
			ka[i] = ka[i+1];
	lu = LUv;
	for(lue = lu + 2*N; lu < lue; lu += 2, l++,u++,varsgn++)
		if (lu[0] <= negInfinity) {
			*u = HUGE_VAL;
			if (lu[1] >= Infinity) {
				*varsgn = 1;
				*l = -HUGE_VAL;
				}
			else {
				*varsgn = -1;
				*l = -lu[1];
				a1 = a + ka[i = l - kp->l];
				ae = a + ka[i+1];
				for(; a1 < ae; a1++)
					*a1 = -*a1;
				}
			}
		else {
			*varsgn = 1;
			*l = lu[0];
			*u = lu[1] >= Infinity ? HUGE_VAL : lu[1];
			}
	if (i = nlogv + niv + nlvbi + nlvci + nlvoi) {
		printf("ignoring integrality of %d variables\n", i);
		need_nl = 0;
		}
	/* qpcheck((long **)&kp->iQ, (long **)&kp->kQ, &kp->Q); */
	return 0;
	}

void printmat(
	int m,
	int n,
	int *iA,
	int *kA,
	double *A
)
{
	int i,j,k;
	double M[8][8];

	if (n>8) return;

	for (i=0; i<m; i++) { 
	    for (j=0; j<n; j++) {
		M[i][j] = 0;
	    }
	}
	for (j=0; j<n; j++) {
	    for (k=kA[j]; k<kA[j+1]; k++) {
		M[iA[k]][j] = A[k];
	    }
	}
	for (i=0; i<m; i++) { 
	    for (j=0; j<n; j++) {
		printf("%7.2f ", M[i][j]);
	    }
	    printf("\n");
	}
	printf("\n");
}

 void
#ifdef KR_headers
amplout(kp, status) LP *kp;
#else
amplout(LP *kp, int status)
#endif
{
	char buf[32], hbuf[256];
	int i;
	double sign, *y, *ye;
	static char *statmsg[] = {
		"optimal solution",	/* 0 */
		"primal unbounded",	/* 1 */
		"primal infeasible",	/* 2 */
		"infinite lower bounds - not implemented",	/* 3 */
		"dual infeasible",	/* 4 */
		"primal and/or dual infeasible",		/* 5 */
		"primal infeasible -- inconsistent equations",	/* 6 */
		"suboptimal solution",	/* 7 */
		"dual unbounded",	/* 8 */
		"resource limit",	/* 9 */
		"??? Solver bug"
		};

	if (status < 0 || status > 9)
		status = 9;
	i = Sprintf(hbuf, "%s: %s", progname, statmsg[status]);
	sign = 1.;
	if (kp->max < 0) {
		sign = -1.;
		for(y = kp->y, ye = y + kp->m; y < ye; y++)
			*y = -*y;
		}
	if (status < 3) {
		g_fmtop(buf, kp->primal_obj);
		i += Sprintf(hbuf+i, "\nprimal objective %s", buf);
		g_fmtop(buf, kp->dual_obj);
		i += Sprintf(hbuf+i, "\n  dual objective %s", buf);
		}
	write_soln(hbuf, kp->x, kp->y);
	}

 static double Times[4];

 static void
show_times(VOID)
{
	int i;

	Times[3] = xectim_();
	for(i = 1; i <= 2; i++)
	  if (time_flag & i) {
		fprintf(i == 1 ? stdout : stderr,
		"\nTimes (seconds):\nInput = %lf\nSolve = %lf\nOutput = %lf\n", 
			Times[1] - Times[0], 
			Times[2] - Times[1], 
			Times[3] - Times[2]);
		}
	}

 void
#ifdef KR_headers
update_kp(kp) LP *kp;
#else
update_kp(LP *kp)
#endif
{
	kp->inftol = opns.iinftol;
        kp->inftol2 = opns.inftol2;
	kp->max = opns.maxflag ? opns.maxflag : 1;
        kp->itnlim = opns.itnlim;
        kp->timlim = opns.timlim;
        kp->epssol = opns.epssol;
        kp->epsnum = opns.epsnum;
        kp->epscdn = opns.epscdn;
        kp->stablty = opns.stablty;
        kp->method = opns.method;
        kp->dense = opns.dense;
        kp->pdf = opns.pdf;
        kp->sf_req = opns.sf_req;
        kp->verbose = opns.v;
	strcpy(kp->obj, opns.obj);
	strcpy(kp->rhs, opns.rhs);
	strcpy(kp->ranges, opns.ranges);
	strcpy(kp->bounds, opns.bounds);
}

 static int
#ifdef KR_headers
env_options(argv) char **argv;
#else
env_options(char **argv)
#endif
{
	char **av0, **av1, *s, *s1=NULL, str[20];

	strcpy(str, progname);
	strcat(str, "_options");
	if(s = getenv(str))
		for(; *s; s = s1)
			s1 = get_option(s);
	for(av0 = av1 = argv; s = *argv; argv++)
		if (strchr(s,'='))
			for(; *s; s = s1)
				s1 = get_option(s);
		else
			*av1++ = s;
	if (n_badopts)
		exit(1);
	*av1 = 0;
	return av1 - av0;
	}

void
#ifdef KR_headers
amplinterface(argc, argv) char **argv;
#else
amplinterface(int argc, char **argv)
#endif
{
	int	namelen, status;
	char	*av[2];	
	LP	*kp, *openlp();

	progname = basename(*argv);
	kp = openlp();
	if (kp == NULL) {
		fprintf(stderr,"Bug: openlp failure!\n"); exit(1);
	}

	argc--; argv++;
	Times[0] = xectim_();
	need_nl = printf("%s: ", progname);
	env_options(argv+2);
	fflush(stdout);
	if (amplin(*argv,kp)) {
		namelen = stub_end - filename;
		av[0] = (char *)Malloc(namelen + 5);
		strncpy(av[0], filename, namelen);
		strcpy(av[0] + namelen, ".spc");
		strcpy(stub_end, ".mps");
		av[1] = filename;
		readlp(2,av,kp);
		}
	update_kp(kp);
	Times[1] = xectim_();
	
	/* print_kp(kp);  */

	status = solvelp(kp);

	Times[2] = xectim_();
	amplout(kp, status);
	show_times();
}

void print_kp( LP *kp )
{
	int i,j,k;

	printf("kQ: ");
	for (j=0; j<kp->n+1; j++) printf("%10d",kp->kQ[j]);
	printf("\n");

	printf("iQ: ");
	for (k=0; k<kp->qnz; k++) printf("%10d",kp->iQ[k]);
	printf("\n");

	printf("Q: ");
	for (k=0; k<kp->qnz; k++) printf("%10.5f",kp->Q[k]);
	printf("\n");

	printf("kA: ");
	for (j=0; j<kp->n+1; j++) printf("%10d",kp->kA[j]);
	printf("\n");

	printf("iA: ");
	for (k=0; k<kp->nz; k++) printf("%10d",kp->iA[k]);
	printf("\n");

	printf("A: ");
	for (k=0; k<kp->nz; k++) printf("%10.5f",kp->A[k]);
	printf("\n");

	printf("b: ");
	for (i=0; i<kp->m; i++) printf("%10.5f",kp->b[i]);
	printf("\n");

	printf("c: ");
	for (j=0; j<kp->n; j++) printf("%10.5f",kp->c[j]);
	printf("\n");
}
