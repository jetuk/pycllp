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

#define NLP_H_included

#define Malloc(x) mymalloc((fint)(x))
#define Realloc(x,y) myralloc(x,(fint)(y))
#define Sizeof(x) (fint)sizeof(x)

#ifndef real
#define real double
#endif

#ifndef Long
#define Long long
#endif
#ifndef fint
#define fint Long
#endif

#ifndef FUNCADD_H_included
#include "funcadd.h"
#endif

extern real Infinity, negInfinity;
 typedef struct {jmp_buf jb;} Jmp_buf;

struct	argpair;
struct	cde;
struct	cexp;
struct	cexp1;
struct	cgrad;
struct	cplist;
struct	de;
struct	derp;
union	ei;
struct	expr;
struct	expr_f;
struct	expr_h;
struct	expr_if;
struct	expr_n;
struct	expr_v;
struct	expr_va;
struct	func_info;
struct	funnel;
struct	linpart;
struct	list;
struct	ograd;
struct	plterm;
struct	relo;
#ifndef __cplusplus
typedef struct argpair argpair;
typedef struct cde cde;
typedef struct cexp cexp;
typedef struct cexp1 cexp1;
typedef struct cgrad cgrad;
typedef struct cplist cplist;
typedef struct de de;
typedef struct derp derp;
typedef union  ei ei;
typedef struct expr expr;
typedef struct expr_f expr_f;
typedef struct expr_h expr_h;
typedef struct expr_if expr_if;
typedef struct expr_n expr_n;
typedef struct expr_v expr_v;
typedef struct expr_va expr_va;
typedef struct func_info func_info;
typedef struct funnel funnel;
typedef struct linpart linpart;
typedef struct list list;
typedef struct ograd ograd;
typedef struct plterm plterm;
typedef struct relo relo;
#else
 extern "C" {
#endif
#ifdef KR_headers
#define Const /* */
#define VOID /*void*/
#define ANSI(x) ()
#define Char char
#else
#define Const const
#define VOID void
#define ANSI(x) x
#define Char void
#endif
 extern void *b_search ANSI((void *ow, int owsize, int n, char **sp));
 extern char *basename ANSI((const char*));
 extern void comeval ANSI((int, int));
 extern void com1eval ANSI((int, int));
 extern void funnelset ANSI((funnel *));
 extern void derprop ANSI((derp *));
 extern char *read_line ANSI((FILE *));
 extern void edag_reset ANSI((int));
 extern void edagread ANSI((FILE*));
 extern char *pr_unknown ANSI((FILE*, char*));
 extern void scream ANSI((char *, int, int));
 extern Char *mymalloc ANSI((fint));
 extern Char *myralloc ANSI((void *, fint));
 extern void write_soln ANSI((char *msg, double *x, double *y));
 extern int g_fmt ANSI((char*, double));
 extern int g_fmtp ANSI((char*, double, int));
 extern int g_fmtop ANSI((char*, double));
 extern int  mip_pri ANSI((int **startp, int **nump, int **prip, fint pmax));
 extern int Sprintf ANSI((char*, const char*, ...));
 extern fint qpcheck ANSI((fint **rowqp, fint **colqp, double **delsqp));
 extern void badline(VOID);
 extern int obj_prec(VOID);
 extern void report_where(VOID);
 extern char *bs_blank, g_fmt_E;
 extern int g_fmt_decpt;
 extern real xectim_(VOID);
#ifdef __cplusplus
	}
#endif

typedef real efunc ANSI((expr *));

extern int need_nl, want_derivs;

 struct
plterm {
	int	n;	/* number of slopes */
	real	bs[1];	/* slope 1, bkpt 1, slope 2, bkpt 2, ..., slope n */
	};

 union
ei {
	expr	*e;
	expr	**ep;
	expr_if	*eif;
	expr_n	*en;
	int	i;
	plterm	*p;
	de	*d;
	real	*rp;
	derp	*D;
	cexp	*ce;
	};

 struct
expr {
	efunc *op;
	int a;
	real dL;
	ei L, R;
	real dR;
	};

 struct
expr_n {	/* for numbers */
	efunc *op;
	real v;
	};

 struct
expr_v {
	efunc *op;
	int a;
	real v;
	};

 struct
expr_if {
	efunc *op;
	int a;
	expr *e, *T, *F;
	derp *D, *dT, *dF, *d0;
	ei Tv, Fv;
	expr_if *next;
	};

 struct
expr_va {
	efunc *op;
	int a;
	ei L, R;
	expr_va *next;
	derp *d0;
	};

 struct
derp {
	derp *next;
	ei a, b, c;
	};

 struct
relo {
	relo *next, *next2;
	derp *D, *Dnext, *Dcond;
	};

 struct
cplist {
	cplist	*next;
	ei	ca;
	real	*cfa;
	};

 struct
cde {
	expr	*e;
	derp	*d;
	int	zaplen;
	};

 struct
de {
	expr *e;
	derp *d;
	ei dv;
	};

 struct
cgrad {
	cgrad *next;
	int  varno;
	int  goff;
	real coef;
	};

 struct
ograd {
	ograd *next;
	int  varno;
	real coef;
	};

 struct
linpart {
	ei	v;
	real	fac;
	};

 struct
list {
	list	*next;
	ei	item;
	};

 struct
cexp1 {
	expr	*e;
	int	nlin;
	linpart	*L;
	};

 struct
cexp {
	expr	*e;
	int	nlin;
	linpart	*L;
	funnel	*funneled;
	list	*cref;
	ei	z;
	int	zlen;
	derp	*d;
	int	*vref;
	};

 struct
funnel {
	funnel	*next;
	cexp	*ce;
	derp	*fulld;
	cplist	*cl;
	cde	fcde;
	};

 struct
argpair {
	expr	*e;
	union {
		char	**s;
		real	*v;
		} u;
	};

 struct
expr_f {
	efunc	*op;
	int	a;
	ufunc	*f;
	arglist	*al;
	argpair	*ap, *ape, *sap, *sape;
	expr	*args[1];
	};

 struct
expr_h {
	efunc	*op;
	int	a;
	char	sym[1];
	};

 struct
func_info {
	func_info	*next, *fnext;
	Const char	*name;
	ufunc		*funcp;
	int		ftype;
	int		nargs;
	};

 struct
edag_info {
	cde	*con_de_;	/* constraint deriv. and expr. info */
	cde	*obj_de_;	/* objective  deriv. and expr. info */
	expr_v	*var_e_;	/* variable values (and related items) */
	cgrad	**Cgrad_;	/* constraint gradient info. (linear part) */
	ograd	**Ograd_;	/* objective  gradient info. (linear part) */
	real	*adjoints_;	/* partials of result w.r.t. current oper. */
	real	*adjoints_nv1_;	/* internal use: start of portion to zero */
	real	*LUrhs_,	/* constraint lower (and, if Urhsx == 0, */
				/* upper) bounds */
		*Urhsx_,	/* constraint upper bounds (if nonzero) */
		*X0_,		/* initial guess (if nonzero) */
		*LUv_,		/* variable lower (and, if Uvx == 0, upper) */
				/* bounds */
		*Uvx_,		/* variable upper bounds (if nonzero) */
		*Lastx_,	/* internal use: copy of X */
		*pi0_;		/* dual initial guess */

			/* stuff for "defined" variables */
	funnel	*f_b_;
	funnel	*f_c_;
	funnel	*f_o_;
	expr_v	*var_ex_,
		*var_ex1_;
	cexp	*cexps_;
	cexp1	*cexps1_;

	char	*objtype_;	/* object type array: 0 == min, 1 == max */
	char	*havex0_;	/* if nonzero, havex0_[i] != 0 ==> */
				/* X0_[i] was specified: this lets you */
				/* tell explicit 0's from default 0's */
	char	*havepi0_;	/* analogous to havex0_, but for dual values */
	real	*A_vals_;	/* if nonzero, store linear Jacobian elements */
				/* in A_vals, A_rownos, and A_colstarts */
				/* rather than in Cgrad_ */
				/* for storing linear Jacobian nonzeros */
	int	*A_rownos_,	/* row numbers corresponding to A_vals_ */
		*A_colstarts_;	/* offsets of columns in A_vals_ */
	int	Fortran_;	/* adjustment to A_rownos and A_colstarts */
	int	amax_;		/* number of adjoint cells */

			/* stuff for common expressions (from "defined" vars) */
	int	c_vars_;
	int	comb_;
	int	combc_;
	int	comc1_;
	int	comc_;
	int	como1_;
	int	como_;

	int	lnc_;		/* no. of linear network constraints */
	int	nbv_;		/* no. of linear binary variables */
	int	niv_;		/* no. of linear integer variables */
	int	nlc_;		/* total no. of nonlinear constraints */
	int	nlnc_;		/* no. of nonlinear network constraints */
	int	nlo_;		/* no. of nonlinear objectives */
	int	nlvb_;		/* no. of nonlinear variables in both */
				/* constraints and objectives */
	int	nlvc_;		/* no. of nonlinear variables in constraints */
	int	nlvo_;		/* no. of nonlinear variables in objectives */
				/* nlvc_ and nlvo_ include nlvb_ */
	int	nlvbi_;		/* integer nonlinear variables in both */
				/* constraints and objectives */
	int	nlvci_;		/* integer nonlinear vars just in constraints */
	int	nlvoi_;		/* integer nonlinear vars just in objectives */
	int	nwv_;		/* no. of (linear) network variables (arcs) */
	int	nzc_;		/* no. of nonzeros in constraints' Jacobian */
	int	nzo_;		/* no. of nonzeros in all objective gradients */
	int	n_con_;		/* total no. of constraints */
	int	n_obj_;		/* total no. of objectives */
	int	n_var_;		/* total no. of variables */

			/* internal stuff */

	int	ncom0_;
	int	ncom1_;
	int	nderps_;
	int	nfunc_;
	int	nzjac_;
	int	o_vars_;
	int	want_deriv_;
	int	x0kind_;
	size_t	x0len_;

	char	*filename_;	/* stub + current extension */
	char	*stub_end_;	/* copy new extension (starting with ".") */
				/* here to adjust filename */
	int	binary_nl_;	/* 0 = ASCII format, 1 = binary */
	int	return_nofile_;	/* 0 ==> jacdim0 should exit if stub.nl */
				/* does not exist; 1 ==> return 0 */
	int	plterms_;	/* no. of piecewise-linear terms */
	int	maxrownamelen_;	/* length of longest constraint name */
				/* (if stub.row exists) */
	int	maxcolnamelen_;	/* length of longest constraint name */
				/* (if stub.col exists) */
	int	co_index_;	/* set this to (constraint number - 1) or */
				/* -(objective number) to identify the */
				/* constraint or objective being evaluated */
				/* (used in report_where()) */
	int	cv_index_;	/* used internally */
	int	row_wise_;	/* unused */
	Jmp_buf	*err_jmp_;	/* If nonzero when an error is detected, */
				/* longjmp here (without printing an error */
				/* message). */
	Jmp_buf	*err_jmp1_;	/* If nonzero when an error is detected */
				/* (and err_jmp_ == 0), longjmp here after */
				/* printing an error message. */
	fint	ampl_options_[10];
	fint	obj_no_;	/* objective number (for write_soln and */
				/* read_soln) */
	int	nranges_;	/* no. of ranges (constraints with */
				/* negInfinity < lhs < rhs < Infinity) */
	int	want_xpi0_;	/* & 1 ==> allocate X0_ if an */
				/* initial guess is available */
				/* & 2 ==> allocate pi0_ if a dual */
				/* initial guess is available */

		/* starting subscripts for cexp1's: request by */
		/* assigning these pointers before invoking edagread */

	int	*c_cexp1st_;	/* cexp1 starts for constraints */
	int	*o_cexp1st_;	/* cexp1 starts for objectives */

	unsigned size_expr_n_;	/* size for struct expr_n, for nlc */

	/* extra info for writesol */
	real ampl_vbtol_;

		/* relocated adjoints for common expressions */
		/* (used by nlc; request by allocating) */

	int	**zaC_;	/* for common expressions */
	int	**zac_;	/* for constraints */
	int	**zao_;	/* for objectives */

	/* for nlc */

	int	skip_int_derivs_;

	/* for sparse gradients */

	int	**zerograds_;
	};
typedef struct edag_info edag_info;
extern edag_info edaginfo;
#ifdef KR_headers
extern real f_OPNUM();
#else
extern real f_OPNUM(expr *);
#endif

#define A_colstarts	edaginfo.A_colstarts_
#define A_rownos	edaginfo.A_rownos_
#define A_vals		edaginfo.A_vals_
#define Cgrad		edaginfo.Cgrad_
#define Fortran		edaginfo.Fortran_
#define LUrhs		edaginfo.LUrhs_
#define LUv		edaginfo.LUv_
#define Lastx		edaginfo.Lastx_
#define Ograd		edaginfo.Ograd_
#define Urhsx		edaginfo.Urhsx_
#define Uvx		edaginfo.Uvx_
#define X0		edaginfo.X0_
#define adjoints	edaginfo.adjoints_
#define adjoints_nv1	edaginfo.adjoints_nv1_
#define amax		edaginfo.amax_
#define ampl_options	edaginfo.ampl_options_
#define ampl_vbtol	edaginfo.ampl_vbtol_
#define binary_nl	edaginfo.binary_nl_
#define c_cexp1st	edaginfo.c_cexp1st_
#define c_vars		edaginfo.c_vars_
#define cexps		edaginfo.cexps_
#define cexps1		edaginfo.cexps1_
#define co_index	edaginfo.co_index_
#define comb		edaginfo.comb_
#define combc		edaginfo.combc_
#define comc		edaginfo.comc_
#define comc1		edaginfo.comc1_
#define como		edaginfo.como_
#define como1		edaginfo.como1_
#define con_de		edaginfo.con_de_
#define cv_index	edaginfo.cv_index_
#define err_jmp		edaginfo.err_jmp_
#define err_jmp1	edaginfo.err_jmp1_
#define f_b		edaginfo.f_b_
#define f_c		edaginfo.f_c_
#define f_o		edaginfo.f_o_
#define filename	edaginfo.filename_
#define havepi0		edaginfo.havepi0_
#define havex0		edaginfo.havex0_
#define lnc		edaginfo.lnc_
#define maxcolnamelen	edaginfo.maxcolnamelen_
#define maxrownamelen	edaginfo.maxrownamelen_
#define n_con		edaginfo.n_con_
#define n_obj		edaginfo.n_obj_
#define n_var		edaginfo.n_var_
#define nbv		edaginfo.nbv_
#define ncom0		edaginfo.ncom0_
#define ncom1		edaginfo.ncom1_
#define nderps		edaginfo.nderps_
#define nfunc		edaginfo.nfunc_
#define niv		edaginfo.niv_
#define nlc		edaginfo.nlc_
#define nlnc		edaginfo.nlnc_
#define nlo		edaginfo.nlo_
#define nlogv		edaginfo.nbv_	/* nbv used to be called nlogv */
#define nlvb		edaginfo.nlvb_
#define nlvbi		edaginfo.nlvbi_
#define nlvc		edaginfo.nlvc_
#define nlvci		edaginfo.nlvci_
#define nlvo		edaginfo.nlvo_
#define nlvoi		edaginfo.nlvoi_
#define nranges		edaginfo.nranges_
#define nwv		edaginfo.nwv_
#define nzc		edaginfo.nzc_
#define nzjac		edaginfo.nzjac_
#define nzo		edaginfo.nzo_
#define o_cexp1st	edaginfo.o_cexp1st_
#define o_vars		edaginfo.o_vars_
#define obj_de		edaginfo.obj_de_
#define obj_no		edaginfo.obj_no_
#define objtype		edaginfo.objtype_
#define pi0		edaginfo.pi0_
#define plterms		edaginfo.plterms_
#define return_nofile	edaginfo.return_nofile_
#define row_wise	edaginfo.row_wise_
#define size_expr_n	edaginfo.size_expr_n_
#define skip_int_derivs	edaginfo.skip_int_derivs_
#define stub_end	edaginfo.stub_end_
#define var_e		edaginfo.var_e_
#define var_ex		edaginfo.var_ex_
#define var_ex1		edaginfo.var_ex1_
#define want_xpi0	edaginfo.want_xpi0_
#define want_deriv	edaginfo.want_deriv_
#define x0kind		edaginfo.x0kind_
#define x0len		edaginfo.x0len_
#define zaC		edaginfo.zaC_
#define zac		edaginfo.zac_
#define zao		edaginfo.zao_
#define zerograds	edaginfo.zerograds_
