/*****************************************************************************

                Implementation of the 
		2 phase (Dual then Primal) Simplex Method
		R. Vanderbei, 3 October 1994

Solves problems in the form:

	     T
	max c x

	A x  = b
	  x >= 0

A is an m by N matrix (it is convenient to reserve n for 
the difference N-m).  Artificial variables have been 
added (hence N > m).  One may assume that the last
m columns of A are invertible and hence can be used as
a starting basis.

******************************************************************************/         
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef QuadPrec
#include "Quad.h"
#define double Quad
#else
#define high(x) (x)
#endif

#include "linalg.h"
#include "lu.h"
#include "myalloc.h"
#include "macros.h"

#define EPS  1.0e-8
#define EPS1 1.0e-8
#define EPS2 1.0e-12
#define EPS3 1.0e-12
#define MAX_ITER 1000000

double sdotprod(double *c, double *x_B, int *basics, int m);

void Nt_times_y( 
    int N, 
    double *at, 
    int *iat, 
    int *kat, 
    int *basicflag,
    double *y, 
    int *iy, 
    int ny, 
    double *yN,
    int *iyN,
    int *pnyN
);

int ratio_test(
	double *dy, 
	int   *idy,
	int    ndy,
	double *y 
);

int pick_neg(int m, double *x);

int solver(
    int m,		/* number of constraints */
    int n,		/* number of variables */
    int nz,		/* number of nonzeros in sparse constraint matrix */
    int *ia, 		/* array row indices */
    int *ka, 		/* array of indices into ia and a */
    double *a,		/* array of nonzeros in the constraint matrix */
    double *b, 		/* right-hand side */
    double *c,          /* objective coefficients */
    double  f, 		/* objective function shift */
    double *x, 		/* primal solution (output) */
    double *y,		/* dual solution (output) */
    double *w, 		/* primal slacks (output) */
    double *z		/* dual slacks (output) */
    )
{
    double  *x_B;	/* primal basics */
    double  *y_N;	/* dual nonbasics */

    double  *dy_N;	/*  dual  basics step direction - values (sparse) */
    int    *idy_N;	/*  dual  basics step direction - row indices */
    int     ndy_N;	/* number of nonz in dy_N */

    double  *dx_B;	/* primal basics step direction - values (sparse) */
    int    *idx_B;	/* primal basics step direction - row indices */
    int     ndx_B;	/* number of nonz in dx_B */

    double  *at;	/* sparse data structure for A^T */
    int    *iat;
    int    *kat;

    int     *basics;	/* list of basic variable indices */
    int     *nonbasics;	/* list of non-basic variable indices */
    int     *basicflag; /* neg if nonbasic, pos if basic */

    int     col_in;	/* entering column; index in 'nonbasics' */
    int     col_out;	/* leaving column; index in 'basics' */

    int     iter = 0;	/* number of iterations */
    int     i,j,k,N,v=0;

    double  s, t;
    float   primal_obj;

    double  *vec;
    int    *ivec;
    int     nvec;

    int	    from_scratch;
    char    cr = 13;    /* carriage return */
		
    /*******************************************************************
    * For convenience, we put...
    *******************************************************************/

    N = n+m;

    /*******************************************************************
    * Add slack variables.  We assume the calling routine allocated
    * enough memory.
    *******************************************************************/

    i = 0;
    k = ka[n];
    for (j=n; j<N; j++) {
	c[j]  = 0.0;
	a[k] = 1.0;
	ia[k] = i;
	i++;
	k++;
	ka[j+1] = k;
    }
    nz = k;

    /*******************************************************************
    * Read in the Data and initialize the common memory sites.
    *******************************************************************/

    MALLOC(    x_B, m, double );      
    MALLOC(   dx_B, m, double );      
    MALLOC(    y_N, n, double );      
    MALLOC(   dy_N, n, double );      
    MALLOC(    vec, m, double );
    MALLOC( nonbasics, n, int );  
    MALLOC(    basics, m, int );  
    MALLOC( basicflag, N, int );  
    MALLOC(  idx_B, m,    int );      
    MALLOC(  idy_N, n,    int );      
    MALLOC(   ivec, m,    int );
    MALLOC(     at, nz,  double );
    MALLOC(    iat, nz,   int );
    MALLOC(    kat, m+1,  int );

    /**************************************************************** 
    *  Initialization.              				    *
    ****************************************************************/

    atnum(m,N,ka,ia,a,kat,iat,at);

    for (j=0; j<n; j++) {
	nonbasics[j] = j;
	basicflag[j] = -j-1;
	      y_N[j] = MAX(c[j],1.0);  /* to force dual feasibility */
	      y_N[j] += drand48();     /* to ensure nondegeneracy   */
    }

    for (i=0; i<m; i++) {
	    basics[i] = n+i;
       basicflag[n+i] = i;
	       x_B[i] = b[i];
    }

    lufac( m, ka, ia, a, basics, 0 );

    dbsolve( m, x_B ); /* could be done explicitly in terms of bounds/ranges */

    /****************************************************************
    * 	Begin Phase I (i.e., dual simplex method)
    ****************************************************************/

    printf ("m = %d,n = %d,nz = %d\n",m,N,nz);
    printf(
"---------------------------------------------------------------------------\n"
"          |   Primal      |        |                           arithmetic  \n"
"  Iter    |  Obj Value    |   mu   |   nonz(L)     nonz(U)     operations  \n"
"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n"
    );
    fflush(stdout);

    /****************************************************************
    * 	Main loop                                                   *
    ****************************************************************/

    for (iter=0; iter<MAX_ITER; iter++) {

        /*************************************************************
        * STEP 6: Print statistics and factor/refactor               *
        *************************************************************/
  
        primal_obj = sdotprod(c,x_B,basics,m) + f;
        printf("%8d   %14.7e     NA    ", iter, high(primal_obj) );
	fflush(stdout);
  
        /*************************************************************
        * STEP 1: Pick most negative basic primal                    *
        *************************************************************/

        col_out = pick_neg(m, x_B);
        if ( col_out == -1 ) break;	/* ready for Phase II */

        /*************************************************************
	*                          -1  T                             *
	* STEP 2: Compute dy  = -(B  N) e                            * 
	*                   N            i			     *
	*         where i = col_out                                  *
        *************************************************************/

	vec[0] = -1.0;
	ivec[0] = col_out;
	nvec = 1;

	btsolve( m, vec, ivec, &nvec );  		

	Nt_times_y( N, at, iat, kat, basicflag, vec, ivec, nvec, 
		     dy_N, idy_N, &ndy_N );

        /*************************************************************
	* STEP 3: Ratio test to find entering column                 * 
        *************************************************************/

	col_in = ratio_test( dy_N, idy_N, ndy_N, y_N );

	if (col_in == -1) return 2; 	/* INFEASIBLE */

        /*************************************************************
	*                        -1                                  *
	* STEP 4: Compute dx  = B  N e                               * 
	*                   B         j                              *
	*                                                            *
        *************************************************************/

	j = nonbasics[col_in];
	for (i=0, k=ka[j]; k<ka[j+1]; i++, k++) {
	     dx_B[i] =  a[k];
	    idx_B[i] = ia[k];
	}
	ndx_B = i;

	bsolve( m, dx_B, idx_B, &ndx_B );

        /*************************************************************
        *                                                            *
        * STEP 5: Put       t = x /dx                                *
        *                        i   i                               *
        *                   s = y /dy                                *
        *                        j   j                               *
        *************************************************************/
  
        /* this is inefficient - it should be fixed */
        for (k=0; k<ndx_B; k++) if (idx_B[k] == col_out) break;
  
        t = x_B[col_out]/dx_B[k];
  
        /* this is inefficient - it should be fixed */
        for (k=0; k<ndy_N; k++) if (idy_N[k] == col_in) break;
  
        s = y_N[col_in]/dy_N[k];
  
        /*************************************************************
        * STEP 7: Set y  = y  - s dy                                 *
        *              N    N       N                                *
        *                                                            *
        *             y  = s                                         *
        *              i                                             *
        *                                                            *
        *             x  = x  - t dx                                 *
        *              B    B       B                                *
        *                                                            *
        *             x  = t                                         *
        *              j                                             *
        *************************************************************/
  
        for (k=0; k<ndy_N; k++) {
		j = idy_N[k];
		y_N[j] -= s*dy_N[k];
        }
  
        y_N[col_in] = s;
  
        for (k=0; k<ndx_B; k++) {
		i = idx_B[k];
		x_B[i] -= t*dx_B[k];
        }
  
        x_B[col_out] = t;
  
        /*************************************************************
        * STEP 8: Update basis                                       * 
        *************************************************************/
  
        i =    basics[col_out];
        j = nonbasics[col_in];
        basics[col_out]   = j;
        nonbasics[col_in] = i;
        basicflag[i] = -col_in-1;
        basicflag[j] = col_out;
  
        from_scratch = refactor( m, ka, ia, a, basics, col_out, v );

	if (from_scratch) {printf("\n");} else {printf("%c", cr);}

    } /* End of Phase I */

    printf("End of Phase I \n");

    /****************************************************************
    *   Restore objective function by setting                       *
    *                -1  T                                          *
    *         y  = (B  N) c  - c                                    *
    *          N           B    N                                   *
    ****************************************************************/

    nvec = 0;
    for (i=0; i<m; i++) {
	if ( ABS(c[basics[i]]) > EPS ) {
	     vec[nvec] = c[basics[i]];
	    ivec[nvec] = i;
	    nvec++;
	}
    }

    btsolve( m, vec, ivec, &nvec );

    Nt_times_y( N, at, iat, kat, basicflag, vec, ivec, nvec, 
		dy_N, idy_N, &ndy_N );  /* use sparse vector dy_N */
					/* temporarily */

    for (j=0; j<n; j++) y_N[j] = -c[nonbasics[j]];
    for (k=0; k<ndy_N; k++) {
	j = idy_N[k];
	y_N[j] += dy_N[k];
    }

    /****************************************************************
    *   Begin Phase II (I.e., primal simplex method)                *
    ****************************************************************/

    for (      ; iter<MAX_ITER; iter++) {

        /*************************************************************
        * STEP 6: Print statistics and factor/refactor               *
        *************************************************************/
  
        primal_obj = sdotprod(c,x_B,basics,m) + f;
        printf("%8d   %14.7e     NA    ", iter, high(primal_obj) );
	fflush(stdout);
  
        /*************************************************************
        * STEP 1: Pick most negative nonbasic dual                   *
        *************************************************************/

        col_in = pick_neg(n, y_N);
        if ( col_in == -1 ) break;	/* optimal */

        /*************************************************************
	*                        -1                                  *
	* STEP 2: Compute dx  = B  N e                               * 
	*                   B         j                              *
	*         where j = col_in                                   *
        *************************************************************/

	j = nonbasics[col_in];
	for (i=0, k=ka[j]; k<ka[j+1]; i++, k++) {
	     dx_B[i] =  a[k];
	    idx_B[i] = ia[k];
	}
	ndx_B = i;

	bsolve( m, dx_B, idx_B, &ndx_B );

        /*************************************************************
	* STEP 3: Ratio test to find leaving column                  * 
        *************************************************************/

	col_out = ratio_test( dx_B, idx_B, ndx_B, x_B );

	if (col_out == -1) return 1;	/* UNBOUNDED */

        /*************************************************************
	*                          -1  T                             *
	* STEP 4: Compute dy  = -(B  N) e                            * 
	*                   N            i			     *
	*         where i = col_out                                  *
        *************************************************************/

	 vec[0] = -1.0;
	ivec[0] = col_out;
	nvec = 1;

	btsolve( m, vec, ivec, &nvec );  		

	Nt_times_y( N, at, iat, kat, basicflag, vec, ivec, nvec, 
		     dy_N, idy_N, &ndy_N );

        /*************************************************************
        * STEP 5: Put       t = x /dx                                *
        *                        i   i                               *
        *                   s = y /dy                                *
        *                        j   j                               *
        *************************************************************/
  
        /* this is inefficient - it should be fixed */
        for (k=0; k<ndx_B; k++) if (idx_B[k] == col_out) break;
  
        t    =    x_B[col_out]/dx_B[k];
  
        /* this is inefficient - it should be fixed */
        for (k=0; k<ndy_N; k++) if (idy_N[k] == col_in) break;
  
        s    =    y_N[col_in]/dy_N[k];
  
        /*************************************************************
        * STEP 7: Set y  = y  - s dy                                 *
        *              N    N       N                                *
        *                                                            *
        *             y  = s                                         *
        *              i                                             *
        *                                                            *
        *             x  = x  - t dx                                 *
        *              B    B       B                                *
        *                                                            *
        *             x  = t                                         *
        *              j                                             *
        *************************************************************/
  
        for (k=0; k<ndy_N; k++) {
		j = idy_N[k];
		y_N[j] -= s*dy_N[k];
        }
  
        y_N[col_in] = s;
  
        for (k=0; k<ndx_B; k++) {
		i = idx_B[k];
		x_B[i] -= t*dx_B[k];
        }
  
        x_B[col_out] = t;
  
        /*************************************************************
        * STEP 8: Update basis                                       * 
        *************************************************************/
  
        i =    basics[col_out];
        j = nonbasics[col_in];
        basics[col_out]   = j;
        nonbasics[col_in] = i;
        basicflag[i] = -col_in-1;
        basicflag[j] = col_out;
  
        from_scratch = refactor( m, ka, ia, a, basics, col_out, v );

	if (from_scratch) {printf("\n");} else {printf("%c", cr);}

    }   /* End of Phase II */

    primal_obj = sdotprod(c,x_B,basics,m) + f;
    printf("%8d   %14.7e     NA    \n", iter, high(primal_obj) );
    printf("End of Phase II \n");

    /****************************************************************
    * 	Transcribe solution to x vector and dual solution to y      *
    ****************************************************************/

    for (j=0; j<N; j++) x[j] = 0.0;
    for (i=0; i<m; i++) x[basics[i]] = x_B[i];

    for (j=0; j<N; j++) y[j] = 0.0;
    for (i=0; i<n; i++) y[nonbasics[i]] = y_N[i];

    /****************************************************************
    * 	Split out slack variables and shift dual variables.
    ****************************************************************/

    for (j=0; j<n; j++) z[j] = y[j];
    for (i=0; i<m; i++) {
	y[i] = y[n+i];
	w[i] = x[n+i];
    }

    /****************************************************************
    * 	Free work space                                             *
    ****************************************************************/

    FREE(  vec );
    FREE( ivec );
    FREE(  x_B );
    FREE(  y_N );
    FREE( dx_B );
    FREE(idx_B );
    FREE( dy_N );
    FREE(idy_N );
    FREE( nonbasics );
    FREE( basics );

    return 0;

}   /* End of solver */

void Display_Solution(int m,int *basics,double *X)
{
	int i;
	
	printf("SOLUTION:\n\n");
	for (i=0;i<m;i++)
		printf("  X[%d] = %lf\n",basics[i], high(X[i]) );
}

void Nt_times_y( 
    int n, 
    double *at, 
    int *iat, 
    int *kat, 
    int *basicflag,
    double *y, 
    int *iy, 
    int ny, 
    double *yN,
    int *iyN,
    int *pnyN
)
{
    int i,j,jj,k,kk;

    static double *a=NULL;
    static int  *tag=NULL;
    static int *link=NULL;
    static int  currtag=1;

    if (  a  == NULL) MALLOC(  a,  n,   double);
    if ( tag == NULL) CALLOC( tag, n,   int);
    if (link == NULL) {CALLOC(link, n+2, int); link++;}

    jj = -1;
    for (k=0; k<ny; k++) {
	i = iy[k];
	for (kk=kat[i]; kk<kat[i+1]; kk++) {
	    j = iat[kk];
	    if (basicflag[j] < 0) {
		if (tag[j] != currtag) {
		    a[j] = 0.0;
		    tag[j] = currtag;
		    link[jj] = j;
		    jj = j;
		}
		a[j] += y[k]*at[kk];
	    }
	}
    }
    link[jj] = n;
    currtag++;

    k = 0;
    for (jj=link[-1]; jj<n; jj=link[jj]) {
	if ( ABS(a[jj]) > EPS1 ) {
             yN[k] = a[jj];
            iyN[k] = -basicflag[jj]-1;
            k++;
	}
    }
    *pnyN = k;
}

int ratio_test(
	double *dy, 
	int   *idy,
	int    ndy,
	double *y
)
{
	int j, jj = -1, k;
	double min = HUGE_VAL;

	for (k=0; k<ndy; k++) {
	    if ( dy[k] > EPS1 ) {
	        j = idy[k];
		if ( y[j]/dy[k] < min ) {
			min = y[j]/dy[k];
			 jj = j;
		}
	    }
	}

	return jj;
}

double sdotprod(double *c, double *x_B, int *basics, int m)
{
	int i;
	double prod = 0.0;

	for (i=0; i<m; i++) { prod += c[basics[i]]*x_B[i]; }

	return prod;
}

int pick_neg(int m, double *x)
{
      int i;
      int col = -1;
      double min = -EPS2;

      for (i=0; i<m; i++) {
		if (x[i] < min) { 
			min = x[i];
			col = i;
		}
      }
      return col;
}
