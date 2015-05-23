#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef QuadPrec
#include "Quad.h"
#define double Quad
#else
#define high(x) (x)
#endif

#include "macros.h" 
#include "linalg.h" 
#include "myalloc.h"
#include "tree.h"
#include "heap.h"
#include "lu.h"

#define EPS    0.0
#define EPSSOL 1.0e-5   /* Zero tolerance for consistent eqns w/dep rows */
#define EPSNUM 1.0e-9
#define NOREORD 0
#define MD  1

struct valind {    /* nonzero entry */
        double d;  /* value */
        int    i;  /* row index */
};
typedef struct valind VALIND;

static  void 	ratnum(int m, int *degL , VALIND **L , int *degLt, VALIND **Lt);
static  void 	cycperm(int start, int end, int *perm, int *iperm);

static  void    Bswap(VALIND *v, int i, int j);

static  int     rank;
static  int     *colperm=NULL, *icolperm=NULL, *rowperm=NULL, *irowperm=NULL;
static  int     *degL=NULL, *degLt=NULL; 
static  VALIND  **L=NULL, **Lt=NULL;
static  int     *degU=NULL, *degUt=NULL; 
static  VALIND  **U=NULL, **Ut=NULL;
static  double  *diag=NULL;
static  double  *newcol=NULL;
static  int     *inewcol=NULL;
static  int     nnewcol=0;
static  int	nr=0;
static	int 	**perm=NULL, **iperm=NULL, **row_list=NULL;
static 	int	*ngauss=NULL;
static	double 	**gauss=NULL;
static	int 	*rows=NULL;
static	int 	*col_outs=NULL;
static	int 	*imaxs=NULL;

static  double 	cumtime =0.0;
static  double 	ocumtime=0.0;

void printout(int idx, int m, int row, int col, VALIND **B, VALIND **Bt, 
	int *degB, int *degBt);
void printout2(int idx, int m, int row, int col, VALIND **B, VALIND **Bt, 
	int *degB, int *degBt);
void printout3 (int m, VALIND **B, int *degB, double *diag, int *iperm);
void printout4 (int m, VALIND **B, int *degB, double *diag);

/*-----------------------------------------------------------------+
| LU factorization.                                                |
| Input:                                                           |
|    m          number of rows (= number of columns)               |
|    kA, iA, A  three array sparse representation of m by n        |
|               matrix A                                           |
|    basis      array of m out of n of the column indices of A     |
|               specify a submatrix B of A                         |
| Output:                                                          |
|    static global variables (only visible to other routines in    |
|    this file:                                                    |
|                                                                  |
|    rank       rank of B                                          |
|    B, degB    ragged array representation of L                   |
|    Bt, degBt  ragged array representation of U transpose         |
|               without its diagonal                               |
|    diag       diagonal entries of U                              |
|    colperm, icolperm, rowperm, irowperm                          |
|               column and row permutations and their inverses    */

void lufac( int m, int *kA, int *iA, double *A, int *basis, int v )
{
        int     kk, kkk, tag, rowdeg, coldeg, 
		row, col, row2, col2, row3, col3;

        int     i, j, k, Bnz, Btnz, okey, deg,
                heapnum, cur, method=MD;

	int 	*hkey=NULL, 
		*heap=NULL, *iheap=NULL, *iwork=NULL, *iwork2=NULL;

	int     *degB=NULL, *degBt=NULL; 
	VALIND  **B=NULL, **Bt=NULL;

        double  narth;

	double starttime, endtime;

	starttime = (double) clock();

        /*---------------------------------------------------------+
        | Set/reset number of refactorizations to 0.              */

        if (nr > 0) {
	    for (j=0; j<nr; j++) {
	 	 perm[j] += col_outs[j];
		iperm[j] += col_outs[j];
	        FREE( perm[j] ); FREE( iperm[j] );
		FREE( row_list[j] ); FREE( gauss[j] );
	    }
	    FREE( rows ); FREE( col_outs ); FREE( imaxs );
	    FREE( perm ); FREE( iperm );
	    FREE( row_list ); FREE( gauss ); FREE( ngauss );
	    nr = 0;
	}

        /*---------------------------------------------------------+
        | allocate space for perm, iperm, and diag.               */

        if (colperm == NULL)  { MALLOC( colperm,  m, int ); }
	else                 { REALLOC( colperm,  m, int ); }
	if (icolperm == NULL) { MALLOC( icolperm, m, int ); }
	else		     { REALLOC( icolperm, m, int ); }
	if (rowperm == NULL)  { MALLOC( rowperm,  m, int ); }
	else		     { REALLOC( rowperm,  m, int ); }
	if (irowperm == NULL) { MALLOC( irowperm, m, int ); }
	else		     { REALLOC( irowperm, m, int ); }
	if (diag     == NULL) { MALLOC( diag,     m, double ); }
	else		     { REALLOC( diag,     m, double ); }

        /*---------------------------------------------------------+
        | allocate space for work arrays.                         */

        MALLOC( degB,    m, int   );
        MALLOC( degBt,   m, int   );
        MALLOC( hkey,    m, int   );
        MALLOC( heap,    m, int   );
        MALLOC( iheap,   m, int   );
        MALLOC( iwork,   m, int   );
        MALLOC( iwork2,  m, int   );

        heap--;         /* so that indexing starts from 1 */

        /*---------------------------------------------------------+
        | calculate degrees in B and Bt                           */

        for (i=0; i<m; i++) { degBt[i] = 0; }
        for (i=0; i<m; i++) {
                degB[i] = kA[ basis[i]+1 ] - kA[ basis[i] ];
                for (k=kA[ basis[i] ]; k<kA[ basis[i]+1 ]; k++) {
                        degBt[ iA[k] ]++;
                }
        }

        /*---------------------------------------------------------+
        | initialize B and Bt                                     */

        MALLOC( B,  m, VALIND * );
        MALLOC( Bt, m, VALIND * );
        for (i=0; i<m; i++) {
		B[i] = NULL;
		Bt[i] = NULL;
                MALLOC( B[i],  degB[i],  VALIND );
                MALLOC( Bt[i], degBt[i], VALIND );
        }

        for (i=0; i<m; i++) { iwork[i] = 0; }
        for (j=0; j<m; j++) {
            kkk = 0;
            for (k=kA[ basis[j] ]; k<kA[ basis[j]+1 ]; k++) {
                row = iA[k];
                kk  = iwork[row];
                B[j][kkk].i = row;
                B[j][kkk].d = A[k];
                Bt[row][kk].i = j;
                Bt[row][kk].d = A[k];
                iwork[row]++;
                kkk++;
            }
        }

        /*---------------------------------------------------------+
        | miscellaneous initializations.                          */

        for (i=0; i<m; i++) { 
            icolperm[i] = -1;
            irowperm[i] = -1;
            iwork[i] = 0; 
            iwork2[i] = -1; 
        }

        rank = m; tag = 0; Bnz = 0; Btnz = 0; 

        /*---------------------------------------------------------+
        | hkey encodes the tie-breaking rule - currently the rule  |
        | is somewhat random.  to make it first occuring minimum,  |
        | change the formula to:                                   |
        |       hkey[node] = degree[node]*m + node;                |
        | warning: with this definition of hkey, there is the      |
        | possibility of integer overflow on moderately large      |
        | problems.                                                |
        |                                                         */

        for (j=0; j<m; j++) {
            if (method == MD) hkey[j] = degBt[j];
            /* if (method == MD) hkey[j] = m*degBt[j] + j; */
            else              hkey[j] = j;

            if (method == MD && hkey[j]==0) hkey[j]=m+1;
        }

        /*---------------------------------------------------------+
        | set up heap structure for quickly accessing minimum.    */

        heapnum = m;
        for (j=m-1; j>=0; j--) {
                cur = j+1;
                iheap[j] = cur;
                heap[cur] = j;
                hfall( heapnum, hkey, iheap, heap, cur );
        }

        /*---------------------------------------------------------+
        | the min degree ordering loop                            */

        for (i=0; i<m; i++) {

                /*------------------------------------------------+
                |  select row with min column degree             */

again:
                row    = heap[1];
                rowdeg = degBt[row];

                if (rowdeg == 0) {
                    printf("singular matrix. rank deficiency = %d\n", m-i);
                    rank = i;
                    goto end;
                }

                /*------------------------------------------------+
                |  select pivot element from this row by          |
                |  choosing nonzero whose col is of minimal       |
                |  degree                                        */

		if (method == MD) {
                  coldeg = m+1;
                  for (k=0; k<rowdeg; k++) {
                    if ( degB[ Bt[row][k].i ] < coldeg 
                         && ABS( Bt[row][k].d ) > EPSNUM ) {
                        col    = Bt[row][k].i;
                        coldeg = degB[col];
                    }
                  }
                  if (coldeg == m+1) {
                    hkey[row]=m+2;
                    hfall( heapnum, hkey, iheap, heap, iheap[row] ); 
                    if (hkey[heap[1]] == m+2) {
                        printf("singular matrix. rank deficiency = %d\n", m-i);
                        rank = i;
                        goto end;
                    } else {
                        goto again;
                    }
                  }
		} else {
		  col    = Bt[row][degBt[row]-1].i;
		  coldeg = degB[col];
                  for (k=0; k<rowdeg; k++) {
                    if ( Bt[row][k].i == row ) {
                        col    = row;
                        coldeg = degB[col];
			break;
                    }
                  }
		}

                /*------------------------------------------------+
                |  update permutation information                */

                colperm[i] = col;
                icolperm[col] = i;

                rowperm[i] = row;
                irowperm[row] = i;

                /*------------------------------------------------+
		|  remove eliminated elements from B and Bt      */

                for (k=0; k<coldeg; k++) {
                    row2 = B[col][k].i;
                    for (kk=0; Bt[row2][kk].i != col; kk++) ;

		    if (row2 != row) {
		        degBt[row2]--;
		        Bswap( Bt[row2], degBt[row2], kk );
		    }
                }

                for (k=0; k<rowdeg; k++) {
                    col2 = Bt[row][k].i;
                    for (kk=0; B[col2][kk].i != row; kk++) ;

		    degB[col2]--;
		    Bswap( B[col2], degB[col2],  kk );
                }

                for (kk=0; Bt[row][kk].i != col; kk++) ;
		diag[i] = Bt[row][kk].d;
		degBt[row]--;
		Bswap( Bt[row], degBt[row], kk );

                /*------------------------------------------------+
                |  update heap                                   */

                okey = hkey[col];
                heap[1] = heap[heapnum];
                iheap[heap[1]] = 1;
                heapnum--;
                if (okey < hkey[heap[1]]) 
                        hfall(heapnum, hkey, iheap, heap, 1);

                /*------------------------------------------------+
                |  generate fillin and update elements           */

                for (k=0; k<degB[col]; k++) {
                    row2 = B[col][k].i;
                    tag++;
                    for (kk=0; kk<degBt[row2]; kk++) {
			col2 = Bt[row2][kk].i;
                        iwork[ col2] = tag; /* tag these columns */
                        iwork2[col2] = kk;  /* say where they are */
                    }
                    for (kk=0; kk<degBt[row]; kk++) {
                        col2 = Bt[row][kk].i;
                        if ( iwork[col2] == tag ) {
			    kkk = iwork2[col2];
                            Bt[row2][kkk].d 
			    -= B[col][k].d * Bt[row][kk].d / diag[i];
			    if ( ABS(Bt[row2][kkk].d) <= 1.0e-12) {
				degBt[row2]--;
				col3 = Bt[row2][degBt[row2]].i;
				iwork [col3] = iwork [col2];
				iwork2[col3] = iwork2[col2];
				Bswap( Bt[row2], degBt[row2], kkk );
			    }
                        } else {
                            deg = degBt[row2];
                            REALLOC( Bt[row2], deg+1, VALIND );
                            Bt[row2][deg].i = col2;
                            Bt[row2][deg].d 
			    = -B[col][k].d * Bt[row][kk].d / diag[i];
                            degBt[row2]++;
                        }
                    }
                }

                for (k=0; k<degBt[row]; k++) {
                    col2 = Bt[row][k].i;
                    tag++;
                    for (kk=0; kk<degB[col2]; kk++) {
			row2 = B[col2][kk].i;
                        iwork[ row2] = tag; /* tag these rows */
                        iwork2[row2] = kk;  /* say where they are */
                    }
                    for (kk=0; kk<degB[col]; kk++) {
                        row2 = B[col][kk].i;
                        if ( iwork[row2] == tag ) {
			    kkk = iwork2[row2];
                            B[col2][kkk].d 
			    -= B[col][kk].d * Bt[row][k].d / diag[i];
			    if ( ABS(B[col2][kkk].d) <= 1.0e-12) {
				degB[col2]--;
				row3 = B[col2][degB[col2]].i;
				iwork [row3] = iwork [row2];
				iwork2[row3] = iwork2[row2];
				Bswap( B[col2], degB[col2], kkk );
			    }
                        } else {
                            deg = degB[col2];
                            REALLOC( B[col2], deg+1, VALIND );
                            B[col2][deg].i = row2;
                            B[col2][deg].d 
			    = -B[col][kk].d * Bt[row][k].d/diag[i];
                            degB[col2]++;
                        }
                    }
                }

                /*------------------------------------------------+
                |  adjust heap                                   */

                for (k=0; k<degB[col]; k++) {
                        row2 = B[col][k].i;
                        if (method == MD) {
                                hkey[row2] = degBt[row2];
                                /* hkey[row2] = m*degBt[row2] + row2; */
				if (hkey[row2]==0) hkey[row2]=m+1;
                        } else {
                                hkey[row2] = row2;
                        }
                        hrise( hkey, iheap, heap, iheap[row2] );
                        hfall( heapnum, hkey, iheap, heap, iheap[row2] ); 
                }
        }
end:

        /*------------------------------------------------+
        |  process dependent rows/cols                   */

        i = rank;
        for (col=0; col<m; col++) {
            if (icolperm[col] == -1) {
                colperm[i] = col;
                icolperm[col] = i;
		degB[col] = 0;
                i++;
            }
        }

        i = rank;
        for (row=0; row<m; row++) {
            if (irowperm[row] == -1) {
                rowperm[i] = row;
                irowperm[row] = i;
		degBt[row] = 0;
                i++;
            }
        }

        for (i=rank; i<m; i++) { diag[i] = 0.0; }

        /*------------------------------------------------+
        |  free up space                                 */

        heap++;
        FREE(hkey); FREE(heap); FREE(iheap);
        FREE(iwork); FREE(iwork2); 

        /*------------------------------------------------+
        |  divide each column of L by diagonal           */

        for (col=0; col<m; col++) {
            for (k=0; k<degB[col]; k++) {
		i = icolperm[col];
                B[col][k].d /= diag[i];
            }
        }

        /*---------------------------------------------------------+
        | calculate and print statistics.                         */

        narth = 0.0e0;
        for (i=0; i<m; i++) {
                k = degB[i];  narth += (double) k*k; Bnz  += k;
                k = degBt[i]; narth += (double) k*k; Btnz += k;
        }
        narth = narth + 3*Bnz + 3*Btnz + 2*m;

        if (v) {
                printf("%9d   %9d %15.0f", Bnz, Btnz, narth);
                fflush(stdout);
        }

	if (degL ==NULL) {MALLOC(degL, m,int);  } else {REALLOC(degL,m,int);}
	if (   L ==NULL) {CALLOC(L, m,VALIND *);} else {REALLOC(L,m,VALIND *);}
	if (degUt==NULL) {MALLOC(degUt,m,int);  } else {REALLOC(degUt,m,int); }
	if (   Ut==NULL) {CALLOC(Ut,m,VALIND *);} else {REALLOC(Ut,m,VALIND *);}
	for (i=0; i<m; i++) {
	    col = colperm[i];
	    degL[i] = degB[col];
	    REALLOC( L[i], degL[i], VALIND );
	    for (k=0; k<degL[i]; k++) {
		L[i][k].d =           B[col][k].d;
		L[i][k].i = irowperm[ B[col][k].i ];
	    }
	}
	for (i=0; i<m; i++) {
	    row = rowperm[i];
	    degUt[i] = degBt[row];
	    REALLOC( Ut[i], degUt[i], VALIND );
	    for (k=0; k<degUt[i]; k++) {
		Ut[i][k].d =           Bt[row][k].d;
		Ut[i][k].i = icolperm[ Bt[row][k].i ];
	    }
	}
	for (i=0; i<m; i++) { FREE( B[i] ); FREE( Bt[i] ); }
	FREE( degB  ); FREE( B  );
	FREE( degBt ); FREE( Bt );
	if (degLt==NULL){MALLOC(degLt, m,int);  } else {REALLOC(degLt,m,int);}
	if (   Lt==NULL){CALLOC(Lt, m,VALIND *);} else {REALLOC(Lt,m,VALIND *);}
	if (degU ==NULL){MALLOC(degU,m,int);    } else {REALLOC(degU,m,int); }
	if (   U ==NULL){CALLOC(U,m,VALIND *);  } else {REALLOC(U,m,VALIND *);}

	ratnum(m, degL , L , degLt, Lt);
	ratnum(m, degUt, Ut, degU , U );

	endtime = (double) clock();
	cumtime += endtime - starttime;
}

int refactor(
    int m,
    int *kA,
    int *iA,
    double *A,
    int *basics,
    int col_out,
    int v
)
{
	int i, j, k, kk, kkk, kkkk, imax;

	int changes, row, col, row2, col2, cnt;
	int bumpstart, bumpend;
	int Utnz=0;

	double val;

	static int	*iwork=NULL;
	static double	*dwork=NULL;
	static void	**pwork=NULL;

	static double *y=NULL;
	static int  *tag=NULL;
	static int  currtag=1;

	static int call=0;
	double starttime, endtime;
	double rffactor = 1.0;

	int    from_scratch;

        /*------------------------------------------------------+
        | Check if it is time to refactor from scratch         */

	call++;
	if ( col_out < 0 || call <= 1) {
		ocumtime = 0.0;
		cumtime  = 0.0;
		lufac( m, kA, iA, A, basics, v );
		cumtime  *= rffactor;
		from_scratch = TRUE;
		return from_scratch;
	}
	if ( call > 3 && cumtime/call >= ocumtime/(call-1) ) {
		ocumtime = 0.0;
		cumtime  = 0.0;
		call = 1;
		lufac( m, kA, iA, A, basics, v );
		cumtime  *= rffactor;
		from_scratch = TRUE;
		return from_scratch;
	}

	ocumtime  = cumtime;
	starttime = (double) clock();

        /*------------------------------------------------------+
        | Allocate storage for work arrays                     */

	if (iwork == NULL) MALLOC( iwork, m, int);
        if (dwork == NULL) MALLOC( dwork, m, double);
        if (pwork == NULL) MALLOC( pwork, m, void *);

        /*------------------------------------------------------+
        | Put col_out into `new' indices                       */

	col_out = icolperm[col_out];

        /*------------------------------------------------------+
        | Compute largest row index for new column             */

	if (inewcol == NULL) { 
	    printf("ERROR: refactoring before bsolving \n");
	    exit(0);
	}

	imax=0;
	for (k=0; k<nnewcol; k++) {
	    imax = MAX(imax, inewcol[k]);
	}

	if (imax < col_out) { 
	    printf("singular matrix \n");
	    from_scratch = FALSE;
	    return from_scratch;
	}

        /*------------------------------------------------------+
        | Insert newcol into col_out column of U (and Ut)       |
	|                                                       |
	|             0 1 2 3 4 5                               |
	|                                                       |
	|          0  x * x x x x                               |
	|          1    * x x x x                               |
	|    U  =  2    * x x x x (here col_out=1 and imax=4)   |
	|          3    *   x x x                               |
	|          4    *     x x                               |
	|          5            x                              */

	/* first remove oldcol from Ut */
	for (k=0; k<degU[col_out]; k++) {
	    row = U[col_out][k].i;
	    for (kk=0; kk<degUt[row]; kk++) {
		if (Ut[row][kk].i == col_out) break; /* INEFFICIENT */
	    }
	    if (kk < degUt[row]) {
		degUt[row]--;
		Bswap( Ut[row], degUt[row], kk );
	    }
	}

	degU[col_out] = nnewcol;
	REALLOC( U[col_out], nnewcol, VALIND );
	kkkk = 0;
	diag[col_out] = 0.0;
	for (k=0; k<nnewcol; k++) {
	    row = inewcol[k];
	    val =  newcol[k];
	    if (row != col_out) {
	        U[col_out][kkkk].i = row;
	        U[col_out][kkkk].d = val;
		kkkk++;

		kkk = degUt[row];
		degUt[row]++;
		REALLOC( Ut[row], degUt[row], VALIND );
		Ut[row][kkk].i = col_out;
		Ut[row][kkk].d = val;
	    } else {
		diag[row] = val;
	    }
	}
	degU[col_out] = kkkk;

        /*------------------------------------------------------+
        | Allocate storage for permutation arrays and shift     |
	| so that indexing begins at col_out                   */

	REALLOC( perm, nr+1, int *);
	REALLOC(iperm, nr+1, int *);
	MALLOC(  perm[nr], imax-col_out+1, int );  perm[nr] -= col_out;
	MALLOC( iperm[nr], imax-col_out+1, int ); iperm[nr] -= col_out;
	REALLOC(     rows, nr+1, int );
	REALLOC( col_outs, nr+1, int );
	REALLOC(    imaxs, nr+1, int );

        /*------------------------------------------------------+
        | Initialize permutation arrays so that col_out is      |
	| cyclically permuted to imax.  After permutation:      |
	|                                                       |
	|             0 2 3 4 1 5                               |
	|                                                       |
	|          0  x x x x * x                               |
	|    U  =  2    x x x * x (here col_out=1 and imax=4)   |
	|          3      x x * x                               |
	|          4        x * x                               |
	|          1    x x x * x                               |
	|          5            x                               |
	|                                                      */

	for (j=col_out; j<imax; j++) {
	     perm[nr][j]   = j+1;
	    iperm[nr][j+1] = j;
	}
	 perm[nr][imax]    = col_out;
	iperm[nr][col_out] = imax;

        /*------------------------------------------------------+
        | Look for singleton columns/rows and permute columns   |
	| to upper-left and rows to lower-right position in     |
	| bump.  Don't forget that the diagonal is stored       |
	| separately in diag[] and that this contributes one    |
	| nonzero to each column/row investigated.             */

	bumpstart = col_out;
	bumpend   = imax;
	do {
	    changes = 0;

            /*------------------------------------------------------+
            | First look for columns.                               |
	    |                                                       |
	    |       0 1 2 3 4 5          0 3 1 2 4 5                |
	    |                                                       |
	    |    0  x x x x * x       0  x x x x * x                |
	    |    1    x x   * x       3    x     * x                |
	    |    2      x   * x  -->  1      x x * x                |
	    |    3        x * x       2        x * x                |
	    |    4    x x   * x       4      x x * x                |
	    |    5            x       5            x                |
	    |                                                      */

	    for (j=bumpstart; j<bumpend; j++) {
		col = perm[nr][j];
		cnt = 0;
		for (k=0; k<degU[col]; k++) {
		    int Ui = U[col][k].i;
		    if (Ui >= col_out && Ui <= imax) {
		        row = iperm[nr][ Ui ];
		        if (bumpstart <= row && row <= bumpend) cnt++;
		    }
		}

		if (cnt == 0) {
		    cycperm(j, bumpstart, perm[nr], iperm[nr]);
		    bumpstart++;
		    changes++;
		} 
	    }

            /*------------------------------------------------------+
            | Now look for rows.                                    |
	    |                                                       |
	    |       0 1 2 3 4 5          0 2 3 4 1 5                |
	    |                                                       |
	    |    0  x x x x * x       0  x x x * x x                |
	    |    1    x       x       2    x x *   x                |
	    |    2      x x * x  -->  3      x *   x                |
	    |    3        x * x       4    x x * x x                |
	    |    4    x x x * x       1          x x                |
	    |    5            x       5            x                |
	    |                                                      */

	    for (i=bumpend-1; i>=bumpstart; i--) {
		row = perm[nr][i];
		cnt = 0;
		for (k=0; k<degUt[row]; k++) {
		    int Uti = Ut[row][k].i;
		    if (Uti >= col_out && Uti <= imax) {
			col = iperm[nr][ Uti ];
			if (bumpstart <= col && col <= bumpend) cnt++;
		    }
		}

		if (cnt == 0) {
		    cycperm(i, bumpend, perm[nr], iperm[nr]);
		    bumpend--;
		    changes++;
		} 
	    }
	} while (changes > 0);

        /*------------------------------------------------------+
        | Permute rows/columns of U and Ut.                    */

        /*------------------------------------------------------+
        | Permute columns of U and diag.                       */

	for (j=col_out; j<=imax; j++) { 
	    dwork[j] = diag[j]; 
	    iwork[j] = degU[j]; 
	    pwork[j] = (void *)U[j]; 
	}
	for (j=col_out; j<=imax; j++) { 
	    diag[j]  = dwork[perm[nr][j]]; 
	    degU[j]  = iwork[perm[nr][j]]; 
	    U[j]     = (VALIND *)pwork[perm[nr][j]]; 
	}

        /*------------------------------------------------------+
        | Permute rows of U.                                   */

	for (j=col_out; j<m; j++) {
	    for (k=0; k<degU[j]; k++) {
		row = U[j][k].i;
		if (col_out <= row && row <= imax) U[j][k].i = iperm[nr][row];
	    }
	}

        /*------------------------------------------------------+
        | Permute rows of Ut.                                  */

	for (i=col_out; i<=imax; i++) { 
	    iwork[i] = degUt[i]; 
	    pwork[i] = (void *)Ut[i]; 
	}
	for (i=col_out; i<=imax; i++) { 
	    degUt[i]  = iwork[perm[nr][i]]; 
	    Ut[i]     = (VALIND *)pwork[perm[nr][i]]; 
	}

        /*------------------------------------------------------+
        | Permute columns of Ut.                               */

	for (i=0; i<=imax; i++) {
	    for (k=0; k<degUt[i]; k++) {
		col = Ut[i][k].i;
		if (col_out <= col && col <= imax) Ut[i][k].i = iperm[nr][col];
	    }
	}

        /*------------------------------------------------------+
        | Record bump row for later use.                       */

	row          = bumpend;
	rows[nr]     = row;
	col_outs[nr] = col_out;
	imaxs[nr]    = imax;

	if (   y == NULL ) CALLOC(   y, m, double );
	if ( tag == NULL ) CALLOC( tag, m, int );

        /*------------------------------------------------------+
        | Scatter bump row into a dense vector.                */

	for (k=0; k<degUt[row]; k++) {
	    col = Ut[row][k].i;
	    y[col] = Ut[row][k].d;
	    tag[col] = currtag;
	    addtree(col);
	}
	y[row] = diag[row];
	tag[row] = currtag;
	addtree(row);

        /*------------------------------------------------------+
        | Remove bump row from U.                              */

	for (k=0; k<degUt[row]; k++) {
	    col = Ut[row][k].i;
	    for (kk=0; kk<degU[col]; kk++) {
		if (U[col][kk].i == row) break;   /* INEFFICIENT */
	    }
	    if (kk < degU[col]) {
		degU[col]--;
		Bswap(U[col], degU[col], kk);
	    }
	}

        /*------------------------------------------------------+
        | Do Gaussian elimination on scatter vector.           */

	REALLOC( row_list, nr+1, int * );
	MALLOC(  row_list[nr], m, int );
	REALLOC(ngauss, nr+1, int );
	REALLOC( gauss, nr+1, double * );
	MALLOC(  gauss[nr], m, double );

	k=0;
	for (col=getfirst(); col<bumpend; col=getnext()) {
	    row2 = col;
	    row_list[nr][k] = row2;
	    gauss[nr][k] = y[col] / diag[row2];
	    for (kk=0; kk<degUt[row2]; kk++) {
		col2 = Ut[row2][kk].i;
		if (tag[col2] != currtag) {
		    y[col2] = 0.0;
		    tag[col2] = currtag;
		    addtree(col2);
		} 
		y[col2] -= gauss[nr][k] * Ut[row2][kk].d;
	    }
	    k++;
	}
	if (col != bumpend) printf("ERROR: col != bumpend \n");
	ngauss[nr] = k;
	REALLOC(  gauss[nr], k, double );
	REALLOC(  row_list[nr], k, int );

        /*------------------------------------------------------+
        | Add eliminated row to U.  kk counts nonzeros in       |
	| eliminated row.                                      */

	diag[col] = y[col];
	kk = 0;
	for (col=getnext(); col != -1; col=getnext()) {
	    if ( ABS(y[col])>EPS ) {
		k = degU[col];
		degU[col]++;
		REALLOC( U[col], degU[col], VALIND );
		U[col][k].i = row;
		U[col][k].d = y[col];
		kk++;
	    }
	}

	REALLOC( Ut[row], kk, VALIND );

        /*------------------------------------------------------+
        | Remove bump row from Ut and replace with eliminated   |
	| row.                                                 */

	k = 0;
	for (col=getfirst(); col != -1; col=getnext()) {
	    if ( col>bumpend && ABS(y[col]) > EPS ) {
		Ut[row][k].d = y[col];
		Ut[row][k].i = col;
		k++;
	    }
	}
	degUt[row] = k;

	if (k != kk) printf("ERROR: alloc'ed wrong size for Ut\n");

	currtag++;
	killtree();

        /*------------------------------------------------------+
	| Apply permutation to colperm and icolperm            */

	for (j=col_out; j<=imax; j++) { iwork[j] = colperm[j]; }
	for (j=col_out; j<=imax; j++) {icolperm[ colperm[ perm[nr][j]]] = j;}
	for (j=col_out; j<=imax; j++) { colperm[iperm[nr][j]] = iwork[j];}

        /*------------------------------------------------------+
	| Increment number of refactorizations.                */

	nr++;

        for (i=0; i<m; i++) {
                k = degUt[i]; Utnz += k;
        }

        if (v) {
                printf("            %9d ", Utnz);
                fflush(stdout);
        }

	endtime = (double) clock();
	cumtime += endtime - starttime;

	from_scratch = FALSE;
	return from_scratch;
}

static void cycperm(int start, int end, int *perm, int *iperm) 
{
	int j, k;

	if (start < end) {
	    k = perm[start];
	    for (j=start; j<end; j++) {
		perm[j] = perm[j+1];
	    }
	    perm[end] = k;
	    for (j=start; j<=end; j++) {
	        iperm[perm[j]] = j;
	    }
	} else if (start > end) {
	    k = perm[start];
	    for (j=start; j>end; j--) {
		perm[j] = perm[j-1];
	    }
	    perm[end] = k;
	    for (j=start; j>=end; j--) {
	        iperm[perm[j]] = j;
	    }
	}
}

/*-----------------------------------------------------------------+
| Forward/backward solve using LU factorization                    |
| Input:                                                           |
|    m          dimension of array y                               |
|    y          array containing right-hand side                   |
|                                                                  |
|    static global variables (assumed setup by lufac()):           |
|                                                                  |
|    rank       rank of B                                          |
|    L, degL    ragged array representation of L                   |
|    Ut, degUt  ragged array representation of U transpose         |
|               without its diagonal                               |
|    diag       diagonal entries of U                              |
|    colperm, icolperm, rowperm, irowperm                          |
|               column and row permutations and their inverses     |
| Output:                                                          |
|                                          -1                      |
|    y          array containing solution B  y                     |
|                                                                  |
|    integer flag indicating whether system is consistent         */

int     bsolve(
	int m, 
	double *sy,
	int *iy,
	int *pny
)
{
        int i, j, jr, ny=*pny;
        int k, row, row2, consistent=TRUE;
        double beta;
        double eps;

	static double *y=NULL, *yy=NULL;
	static int    *tag=NULL;
	static int  currtag=1;

	double starttime, endtime;

	starttime = (double) clock();

	if (   y  == NULL) CALLOC(   y, m,   double);
	if (  yy  == NULL) CALLOC(  yy, m,   double);
	if ( tag  == NULL) CALLOC( tag, m,   int);

	if ( newcol == NULL) MALLOC(  newcol, m, double );
	if (inewcol == NULL) MALLOC( inewcol, m, int );

	for (k=0; k<ny; k++) {
	    i = irowperm[iy[k]];
	    y[i] = sy[k];
	    tag[i] = currtag;
	    addtree(i);
	}

        if (rank < m) eps = EPSSOL * maxv(sy,ny);

        /*------------------------------------------------------+
        |               -1                                      |
        |       y  <-  L  y                                    */

        for (i=getfirst(); i < rank && i != -1; i=getnext()) {
                beta = y[i];
                for (k=0; k<degL[i]; k++) {
                        row = L[i][k].i;
			if (tag[row] != currtag) {
			    y[row] = 0.0;
			    tag[row] = currtag;
			    addtree(row);
			}
                        y[row] -= L[i][k].d * beta;
                }
        }

        /*------------------------------------------------------+
        | Apply refactorization row operations.                */

	for (jr=0; jr<nr; jr++) {

            /*--------------------------------------------------+
            | Gather sparse vector.                            */

	    k=0;
            for (j=col_outs[jr]; j<=imaxs[jr]; j++) {
		if (tag[j] == currtag) {
		    sy[k] = y[j];
		    iy[k] =   j;
		    k++;
		    tag[j]--;
		    deltree(j); 
		}
	    }
	    ny = k;

            /*--------------------------------------------------+
            | Scatter and permute.                             */

	    for (k=0; k<ny; k++) {
		i = iperm[jr][iy[k]];
		y[i] = sy[k];
		tag[i] = currtag;
		addtree(i);
	    }

            /*--------------------------------------------------+
            | Apply row operations.                            */

	    row = rows[jr];
	    for (k=0; k<ngauss[jr]; k++) {
		row2 = row_list[jr][k];
		if (tag[row] != currtag) {
		    y[row] = 0.0;
		    tag[row] = currtag;
		    addtree(row);
		}
		if (tag[row2] == currtag) {
		    y[row] -= gauss[jr][k] * y[row2];
		}
	    }

	}

        /*------------------------------------------------------+
	|                                       -1              |
        | Set aside sparse intermediate vector L  P a  for      |
	|                                            j          |
        | refactorization routine.                             */

	nnewcol = 0;
	for (i=getfirst(); i != -1; i=getnext()) {
	    if ( ABS(y[i]) > EPS ) {
	        newcol [nnewcol] = y[i];
	        inewcol[nnewcol] = i;
	        nnewcol++;
	    }
	}

        /*------------------------------------------------------+
        |               -1                                      |
        |       y  <-  U  y                                    */

        for (i=getlast(); i >= rank && i != -1; i=getprev()) {
                if ( ABS( y[i] ) > eps ) consistent = FALSE;
                y[i] = 0.0;
        }
        for ( ; i>=0; i=getprev()) {
                beta = y[i]/diag[i];
                for (k=0; k<degU[i]; k++) {
			row = U[i][k].i;
			if (tag[row] != currtag) {
			    y[row] = 0.0;
			    tag[row] = currtag;
			    addtree(row);
			}
			y[row] -= U[i][k].d * beta;
                }
                y[i] = beta;
        }

	ny = 0;
	for (i=getfirst(); i != -1; i=getnext()) {
	    if ( ABS(y[i]) > EPS ) {
	        sy[ny] = y[i];
	        iy[ny] = colperm[i];
	        ny++;
	    }
	}
	*pny = ny;

	currtag++;
	killtree();

	endtime = (double) clock();
	cumtime += endtime - starttime;

        return consistent;
}

/*-----------------------------------------------------------------+
| Forward/backward solve using LU factorization                    |
| Input:                                                           |
|    m          dimension of array y                               |
|    y          array containing right-hand side                   |
|                                                                  |
|    static global variables (assumed setup by lufac()):           |
|                                                                  |
|    rank       rank of B                                          |
|    L, degL    ragged array representation of L                   |
|    Ut, degUt  ragged array representation of U transpose         |
|               without its diagonal                               |
|    diag       diagonal entries of U                              |
|    colperm, icolperm, rowperm, irowperm                          |
|               column and row permutations and their inverses     |
| Output:                                                          |
|                                          -T                      |
|    y          array containing solution B  y                     |
|                                                                  |
|    integer flag indicating whether system is consistent         */

int     btsolve(
        int m,
        double *sy,
        int *iy,
        int *pny
)
{
        int i, j, ny=*pny;
        int k, jr, row, row2, consistent=TRUE;
        double beta;
        double eps;

        static double *y=NULL;
	static int  *tag=NULL;
	static int  currtag=1;

	double starttime, endtime;

	starttime = (double) clock();

        if (   y == NULL) CALLOC(   y, m, double);
        if ( tag == NULL) CALLOC( tag, m, int);

        for (k=0; k<ny; k++) {
	    i = icolperm[iy[k]];
            y[i] = sy[k];
	    tag[i] = currtag;
	    addtree(i);
        }

        if (rank < m) eps = EPSSOL * maxv(sy,ny);

        /*------------------------------------------------------+
        |               -T                                      |
        |       y  <-  U  y                                    */

        for (i=getfirst(); i < rank && i != -1; i=getnext()) {
                beta = y[i]/diag[i];
                for (k=0; k<degUt[i]; k++) {
                        row = Ut[i][k].i;
			if (tag[row] != currtag) {
			    y[row] = 0.0;
			    tag[row] = currtag;
			    addtree(row);
			}
                        y[row] -= Ut[i][k].d * beta;
                }
                y[i] = beta;
        }

        /*------------------------------------------------------+
        | Apply refactorization row operations.                */

	for (jr=nr-1; jr>=0; jr--) {

            /*--------------------------------------------------+
            | Apply row operations.                            */

	    row = rows[jr];
	    for (k=ngauss[jr]-1; k>=0; k--) {
		row2 = row_list[jr][k];
		if (tag[row2] != currtag) {
		    y[row2] = 0.0;
		    tag[row2] = currtag;
		    addtree(row2);
		}
		if (tag[row] == currtag) {
		    y[row2] -= gauss[jr][k] * y[row];
		}
	    }

            /*--------------------------------------------------+
            | Gather sparse vector.                            */

	    k=0;
            for (j=col_outs[jr]; j<=imaxs[jr]; j++) {
		if (tag[j] == currtag) {
		    sy[k] = y[j];
		    iy[k] =   j;
		    k++;
		    tag[j]--;
		    deltree(j);
		}
	    }
	    ny = k;

            /*--------------------------------------------------+
            | Scatter and permute.                             */

	    for (k=0; k<ny; k++) {
		i = perm[jr][iy[k]];
		y[i] = sy[k];
		tag[i] = currtag;
		addtree(i);
	    }

	}

        /*------------------------------------------------------+
        |               -T                                      |
        |       y  <-  L  y                                    */

        for (i=getlast(); i >= rank && i != -1; i=getprev()) {
                if ( ABS( y[i] ) > eps ) consistent = FALSE;
                y[i] = 0.0;
        }

        for ( ; i>=0; i=getprev()) {
                beta = y[i];
                for (k=0; k<degLt[i]; k++) {
		    row = Lt[i][k].i;
		    if (tag[row] != currtag) {
			y[row] = 0.0;
			tag[row] = currtag;
			addtree(row);
		    }
		    y[row] -= Lt[i][k].d * beta;
                }
        }

	ny = 0;
	for (i=getfirst(); i != -1; i=getnext()) {
	    if ( ABS(y[i]) > EPS ) {
	        sy[ny] = y[i];
	        iy[ny] = rowperm[i];
	        ny++;
	    }
	}
	*pny = ny;

	currtag++;
	killtree();

	endtime = (double) clock();
	cumtime += endtime - starttime;

        return consistent;
}

void lu_clo()
{
        FREE( rowperm ); FREE( irowperm );
        FREE( colperm ); FREE( icolperm );
        FREE( diag );
}

static void ratnum(int m, int *degB, VALIND **B, int *degBt, VALIND **Bt)
{
	int i, j, k;
	int *iwork;

	for (i=0; i<m; i++) { degBt[i] = 0; }

	for (j=0; j<m; j++) {
	    for (k=0; k<degB[j]; k++) {
		i = B[j][k].i;
		degBt[i]++;
	    }
	}

	CALLOC( iwork, m, int );
	for (i=0; i<m; i++) { REALLOC( Bt[i], degBt[i], VALIND ); }

	for (j=0; j<m; j++) {
	    for (k=0; k<degB[j]; k++) {
		i = B[j][k].i;
		Bt[i][ iwork[i] ].i = j;
		Bt[i][ iwork[i] ].d = B[j][k].d;
		iwork[i]++;
	    }
	}

	FREE( iwork );
}

static void Bswap(VALIND *v, int i, int j)
{
        VALIND temp;

        temp = v[i];
        v[i] = v[j];
        v[j] = temp;
}

/*-----------------------------------------------------------------+
| Forward/backward solve using LU factorization                    |
| Input:                                                           |
|    m          dimension of array y                               |
|    y          array containing right-hand side                   |
|                                                                  |
|    static global variables (assumed setup by lufac()):           |
|                                                                  |
|    rank       rank of B                                          |
|    L, degL    ragged array representation of L                   |
|    Ut, degUt  ragged array representation of U transpose         |
|               without its diagonal                               |
|    diag       diagonal entries of U                              |
|    colperm, icolperm, rowperm, irowperm                          |
|               column and row permutations and their inverses     |
| Output:                                                          |
|                                          -1                      |
|    y          array containing solution B  y                     |
|                                                                  |
|    integer flag indicating whether system is consistent         */

int     dbsolve(int m, double *y)
{
        int i;
        int k, row, consistent=TRUE;
        double beta, *dwork;
        double eps;

	double starttime, endtime;

	starttime = (double) clock();

	MALLOC( dwork, m, double );

        if (rank < m) eps = EPSSOL * maxv(y,m);
	for (i=0; i<m; i++) dwork[i] = y[i];
	for (i=0; i<m; i++) y[irowperm[i]] = dwork[i];

        /*------------------------------------------------------+
        |               -1                                      |
        |       y  <-  L  y                                    */

        for (i=0; i<rank; i++) {
                beta = y[i];
                for (k=0; k<degL[i]; k++) {
                        row = L[i][k].i;
                        y[row] -= L[i][k].d * beta;
                }
        }

        /*------------------------------------------------------+
	|                                       -1              |
        | Set aside sparse intermediate vector L  P a  for      |
	|                                            j          |
        | refactorization routine.                             */

	if ( newcol == NULL) MALLOC(  newcol, m, double );
	if (inewcol == NULL) MALLOC( inewcol, m, int );

	nnewcol = 0;
	for (i=0; i<m; i++) {
	    if ( ABS(y[i]) > EPS ) {
	        newcol [nnewcol] = y[i];
	        inewcol[nnewcol] = i;
	        nnewcol++;
	    }
	}

        /*------------------------------------------------------+
        |               -1                                      |
        |       y  <-  U  y                                    */

        for (i=m-1; i>=rank; i--) {
                if ( ABS( y[i] ) > eps ) consistent = FALSE;
                y[i] = 0.0;
        }
        for (i=rank-1; i>=0; i--) {
                beta = y[i]/diag[i];
                for (k=0; k<degU[i]; k++) {
			row = U[i][k].i;
                        y[row] -= U[i][k].d * beta;
                }
                y[i] = beta;
        }

	for (i=0; i<m; i++) dwork[i] = y[i];
	for (i=0; i<m; i++) y[colperm[i]] = dwork[i];

	FREE( dwork );

	endtime = (double) clock();
	cumtime += endtime - starttime;

        return consistent;
}

/*-----------------------------------------------------------------+
| Forward/backward solve using LU factorization                    |
| Input:                                                           |
|    m          dimension of array y                               |
|    y          array containing right-hand side                   |
|                                                                  |
|    static global variables (assumed setup by lufac()):           |
|                                                                  |
|    rank       rank of B                                          |
|    L, degL    ragged array representation of L                   |
|    Ut, degUt  ragged array representation of U transpose         |
|               without its diagonal                               |
|    diag       diagonal entries of U                              |
|    colperm, icolperm, rowperm, irowperm                          |
|               column and row permutations and their inverses     |
| Output:                                                          |
|                                          -T                      |
|    y          array containing solution B  y                     |
|                                                                  |
|    integer flag indicating whether system is consistent         */

int     dbtsolve(int m, double *y)
{
        int i;
        int k, row, consistent=TRUE;
        double beta, *dwork;
        double eps;

	double starttime, endtime;

	starttime = (double) clock();

	MALLOC( dwork, m, double );

        if (rank < m) eps = EPSSOL * maxv(y,m);
	for (i=0; i<m; i++) dwork[i] = y[i];
	for (i=0; i<m; i++) y[icolperm[i]] = dwork[i];

        /*------------------------------------------------------+
        |               -T                                      |
        |       y  <-  U  y                                    */

        for (i=0; i<rank; i++) {
                beta = y[i]/diag[i];
                for (k=0; k<degUt[i]; k++) {
			row = Ut[i][k].i;
                        y[row] -= Ut[i][k].d * beta;
                }
                y[i] = beta;
        }
        for (i=m-1; i>=rank; i--) {
                if ( ABS( y[i] ) > eps ) consistent = FALSE;
                y[i] = 0.0;
        }

        /*------------------------------------------------------+
        |               -T                                      |
        |       y  <-  L  y                                    */

        for (i=rank-1; i>=0; i--) {
                beta = y[i];
                for (k=0; k<degLt[i]; k++) {
                        row = Lt[i][k].i;
                        y[row] -= Lt[i][k].d * beta;
                }
        }

	for (i=0; i<m; i++) dwork[i] = y[i];
	for (i=0; i<m; i++) y[rowperm[i]] = dwork[i];

	FREE( dwork );

	endtime = (double) clock();
	cumtime += endtime - starttime;

        return consistent;
}
