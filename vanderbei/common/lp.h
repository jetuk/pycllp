/*********************************************************************/
/***    Copyright (c) Robert J. Vanderbei, 1994                    ***/
/***    All Rights Reserved                                        ***/
/*********************************************************************/

/* #include <stdio.h> */
#include "macros.h"
/* #include <sigfpe.h> */

#define	HEADER 0
#define	NAME   1
#define	ROWS   2
#define	COLS   3
#define	RHS    4
#define	RNGS   5
#define	BNDS   6
#define	QUADS  7
#define	END    8

#define	UNSET  0
#define	PRIMAL 1
#define	DUAL   2

#define FINITE   0x1
#define INFINITE 0x2
#define UNCONST  0x4
#define FREEVAR   0x1
#define BDD_BELOW 0x2
#define BDD_ABOVE 0x4
#define BOUNDED   0x8

#define	LP_OPEN_MAX 20 /* max #	lp problems open at once */

typedef	struct lp_prob {
	int m;		/* number of rows */
	int n;		/* number of columns */
	int nz;		/* number of nonzeros */
	double *A;	/* pointer to array of nonzero values in A */
	int *iA;	/* pointer to array of corresponding row indices */
	int *kA;	/* pointer to array of indices into A (and iA)
				indicating where each new column of A begins */
	double *b;	/* pointer to array containing right-hand side */
	double *c;	/* pointer to array containing objective function */
	double f;	/* fixed adjustment to objective function */
	double *r;	/* pointer to array containing range vector */
	double *l;	/* pointer to array containing lower bounds */
	double *u;	/* pointer to array containing upper bounds */
	int *varsgn;	/* array indicating which variables were declared to
				be non-positive	*/
	char **rowlab;	/* array of strings containing row labels */
	char **collab;	/* array of strings containing column labels */

	int qnz;	/* number of nonzeros in lower triangle	of Q */
	double *Q;	/* pointer to array of nonzero values of Q */
	int *iQ;	/* pointer to array of corresponding row indices */
	int *kQ;	/* pointer to array of indices into Q (and iQ)
				indicating where each new column of Q begins */

	double *At;	/* pointer to array of nonzero values in At */
	int *iAt;	/* pointer to array of corresponding row indices */
	int *kAt;	/* pointer to array of indices into At (and iAt) */

	int *bndmark;	/* pointer to array of bound marks */
	int *rngmark;	/* pointer to array of range marks */

	double *w;	/* pointer to array containing primal surpluses	*/
	double *x;	/* pointer to array containing primal solution */
	double *y;	/* pointer to array containing dual solution */
	double *z;	/* pointer to array containing dual slacks */
	double *p;	/* pointer to array containing range slacks */
	double *q;	/* pointer to array containing dual range slacks */
	double *s;	/* pointer to array containing dual for	ub slacks */
	double *t;	/* pointer to array containing upper bound slacks */
	double *v;	/* pointer to array containing dual for	range (w) */
	double *g;	/* pointer to array containing lower bound slacks */
	double *ub;	/* pointer to array containing shifted upper bounds */

	int max;	/* max = -1, min = 1 */
	double inftol;	/* infeasibility tolerance */
	double inftol2;	/* infeasibility for stopping rule */
	int sf_req;	/* significant figures requested */
	int itnlim;	/* iteration limit */
	double timlim;	/* time limit */
	int verbose;	/* level of verbosity */
	double epssol;	/* epsilon tolerance in f/b solve */
	double epsnum;	/* epsilon tolerance in num fact */
	double epscdn;	/* epsilon tolerance for conditioning */
	double stablty;	/* mixing factor for stability */
	int method;	/* reordering method */
	int dense;	/* threshold for dense columns/rows */
	int pdf;	/* order to favor primal (ADA^T) or dual (A^TDA) */
	char name[15];	/* string containing problem name */
	char obj[11];	/* string containing objective function	name */
	char rhs[11];	/* string containing right-hand	side name */
	char ranges[11];/* string containing range set name */
	char bounds[11];/* string containing bound set name */

	int *tier;	/* tier for factorization priorities */

	char **param;	/* array of strings containing user parameters */
	int np;		/* number of user parameters */
 
	int  (*stopping_rule)(void *);/* pointer to stopping	rule fcn */
	void (*init_vars)(void *);    /* pointer to initialization fcn */

	void (*h_init)(void *);       /* pointer to initialization hook fcn */
	void (*h_update)(void *);     /* pointer to update hook fcn */
	void (*h_step)(void *);       /* pointer to step hook fcn */

	int    iter;	    /* current iteration number	*/
	double elaptime;    /* elapsed time */
	double pres;	    /* primal residual (i.e. infeasibility) */
	double dres;	    /* dual   residual (i.e. infeasibility) */
	int    sigfig;	    /* significant figures */
	double primal_obj;  /* primal objective	value */
	double dual_obj;    /* dual   objective	value */
} LP;

LP	*openlp(void);

void	readlp(
	int	argc,
	char	*argv[],
	LP	*lp
);

int	solvelp(
	LP	*lp
);

void	closelp(
	LP	*lp
);

int	getparam(
	char *label0
);

void	my_exit(
	int num,
	char *str
);

extern void message();

double sdotprod(
	double	*a,
	int	*ja,
	double  *x,
	int	na
);

double dotprod(		/* inner product between n-vectors x and y */
	double	*x,
	double	*y,
	int	n
);

double dotprodm(	/* inner product between n-vectors x and y */
	double	*x,
	double	*y,
	int	n,
	int	*mask,
	int	val
);

void smx(		/* y = sparse matrix (A,kA,iA) times x */
	int	m,
	int	n,
	double	*A,
	int	*kA,
	int	*iA,
	double	*x,
	double	*y
);

void smtx(		/* y = x times sparse matrix (A,kA,iA) */
	int	m,
	int	n,
	double	*A,
	int	*kA,
	int	*iA,
	double	*x,
	double	*y
);

void atnum(		/* (kAt,iAt,At)	= transpose of (kA,iA,A) */
	int	m,
	int	n,
	int	*kA,
	int	*iA,
	double	*A,
	int	*kAt,
	int	*iAt,
	double	*At
);

double maxv(		/* compute componentwise maximum of n-vector x */
	double *x,
	int	n
);

void	writelp(
	LP	*lp,
	char	*fname
);

void	writesol(
	LP	*lp,
	char	fname[]
);

void inv_num(
	LP 	*lp,	/* pointer to the linear program's data */
        double  *dn,    /* diagonal matrix for upper-left  corner */
        double  *dm     /* diagonal matrix for lower-right corner */
);

int solve(
	LP 	*lp,
	double	*Dn,
	double	*Dm,
        double  *c,
        double  *b
);

int deflt_st_rule(
    void *
);

void deflt_init(
    void *
);

void deflt_hook(
    void *
);

void inv_clo(void);

char *my_strdup(
	char	*s1
);

double cputimer(
);

void nl_init( 
    void *vlp 
);

void nl_update( 
    void *vlp 
);

void nlobjterm( 
        void (*func)( double *z, double *pval, double *grad, double **hessian),                           /* function that computes val, grad, hessian at z */
        int k,             /* number of arguments for func() */
        char **rowlabs,    /* row labels for arugments to func() */
        double *argshifts  /* shifts to the arguments to func() */
);

void nlconstr( 
        void (*func)( double *z, double *pval, double *grad, double **hessian),                           /* function that computes val, grad, hessian at z */
	double rhs,
        int k,             /* number of arguments for func() */
        char **rowlabs,    /* row labels for arugments to func() */
        double *argshifts  /* shifts to the arguments to func() */
);

void nlobjterm0( 
        void (*func)( double *z, double *pval, double *grad, double **hessian),                           /* function that computes val, grad, hessian at z */
        int k,             /* number of arguments for func() */
        char **rowlabs,    /* row labels for arugments to func() */
        double *argshifts  /* shifts to the arguments to func() */
);

void nlconstr0( 
        void (*func)( double *z, double *pval, double *grad, double **hessian),                           /* function that computes val, grad, hessian at z */
	double rhs,
        int k,             /* number of arguments for func() */
        char **rowlabs,    /* row labels for arugments to func() */
        double *argshifts  /* shifts to the arguments to func() */
);
