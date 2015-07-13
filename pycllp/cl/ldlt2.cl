#include <linalg2.h>


#define	TRUE 1
#define	FALSE 0


#define _EPS 1.0e-8
#define _EPSSOL 1.0e-6  /* Zero tolerance for consistent eqns w/dep rows */
#define _EPSNUM 0.0     /* Initial zero tolerance for dependent rows */
#define _EPSCDN 1.0e-12 /* Zero tolerance for ill-conditioning test */
#define _EPSDIAG 1.0e-14 /* diagonal perturbation */
#define _STABLTY 1.0    /* 0=fast/unstable, 1=slow/stable */
#define _NOREORD 0
#define _MD  1
#define _MLF 2
#define _DENSE -1
#define _UNSET 0
#define _PRIMAL 1
#define _DUAL 2

void lltnum(
  int m, int n, int lnz, float _max,  int denwin, int* ndep,
  __global float* diag,
  __global float* perm,
  __global int* iperm,
  __global float* A,
  __global int* iA,
  __global int* kA,
  __global float* At,
  __global int* iAt,
  __global int* kAt,
  __global float* AAt,
  __global int* iAAt,
  __global int* kAAt,
  //__global float* Q,
  //__global int* iQ,
  //__global int* kQ,
  __global float* dn,
  __global float* dm,
  __global float* temp,
  __global int* iwork,
  __global int* mark
)
{
  int    kk, k_end;
  int    i, j, newj, k, k_bgn, row;
	int    sgn_diagi;
	int    m2 = m+n;
  float lij_dj;
  float lij;
  float diagi, maxoffdiag, maxdiag=0.0;
  float epsnum = _EPSNUM;
  float eps = _EPS;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  //int wgid = get_group_id(0);

  // local iwork memory pointers
  __global int *first = iwork;
  __global int *link = iwork+gsize*m2;

        /*------------------------------------------------------+
        |                                                       |
        | the input is a symmetric matrix  A  with lower        |
        |       triangle stored sparsely in                     |
	      |       kAAt[], iAAt[], AAt[] and with the diagonal     |
	      |       stored in dia[].                                |
        | the output is the lower triangular matrix  L          |
        |       stored sparsely in  kAAt,iAAt,AAt  and          |
        |       a diagonal matrix  D  stored in the diag.       |
        |                  t                                    |
        |          A  = LDL                                     |
        |                                                       |
        +------------------------------------------------------*/

        /*------------------------------------------------------+
        | initialize constants                                 */

        for (i=gid; i<gsize*m2; i+=gsize) {
          temp[i] = 0.0;
          first[i] = 0;
          link[i] = -1;
        }

        for (i=gid; i<gsize*m2; i+=gsize) {
                if (fabs(diag[i]) > maxdiag) maxdiag = fabs(diag[i]);
        }

        (*ndep)=0;

        /*------------------------------------------------------+
        | begin main loop - this code is taken from George and  |
        | Liu's book, pg. 155, modified to do LDLt instead      |
        | of LLt factorization.                                */

        for (i=0; i<m2; i++) {
          diagi = diag[gid+i*gsize];
	        sgn_diagi = perm[i] < n ? -1 : 1;
            for (j=link[gid+i*gsize]; j != -1; j=newj) {
                newj = link[gid+j*gsize];
                k = first[gid+j*gsize];
                lij = AAt[gid+k*gsize];
                lij_dj = lij*diag[gid+j*gsize];
                diagi -= lij*lij_dj;
                k_bgn = k+1;
                k_end = kAAt[j+1];
                if (k_bgn < k_end) {
                    first[gid+j*gsize] = k_bgn;
                    row = iAAt[k_bgn];
                    link[gid+j*gsize] = link[gid+row*gsize];
                    link[gid+row*gsize] = j;
                    if (j < denwin) {
                        for (kk=k_bgn; kk<k_end; kk++)
                            temp[gid+iAAt[kk]*gsize] += lij_dj*AAt[gid+kk*gsize];
                    } else {
                        int ptr;
                        ptr = row;
                        for (kk=k_bgn; kk<k_end; kk++) {
                            temp[gid+ptr*gsize] += lij_dj*AAt[gid+kk*gsize];
                            ptr++;
                        }
                    }
                }
            }
            k_bgn = kAAt[i];
            k_end = kAAt[i+1];
            for (kk=k_bgn; kk<k_end; kk++) {
                row = iAAt[kk];
		            AAt[gid+kk*gsize] -= temp[gid+row*gsize];
            }
            if (fabs(diagi) <= epsnum*maxdiag || mark[gid+i*gsize] == FALSE) {
	    /*
            if (sgn_diagi*diagi <= epsnum*maxdiag || mark[i] == FALSE)
	    */
                (*ndep)++;
		maxoffdiag = 0.0;
                for (kk=k_bgn; kk<k_end; kk++) {
		    maxoffdiag = fmax( maxoffdiag, fabs( AAt[gid+kk*gsize] ) );
                }
		if ( maxoffdiag < 1.0e+6*eps ) {
		    mark[gid+i*gsize] = FALSE;
		} else {
		    diagi = sgn_diagi * eps;
		}
            }
	    diag[gid+i*gsize] = diagi;
            if (k_bgn < k_end) {
                first[gid+i*gsize] = k_bgn;
                row = iAAt[k_bgn];
                link[gid+i*gsize] = link[gid+row*gsize];
                link[gid+row*gsize] = i;
                for (kk=k_bgn; kk<k_end; kk++) {
                    row = iAAt[kk];
                    if (mark[gid+i*gsize]) {
                            AAt[gid+kk*gsize] /= diagi;
                    } else {
                            AAt[gid+kk*gsize] = 0.0;
		    }
                    temp[gid+row*gsize] = 0.0;
                }
            }
        }

}

void inv_num(
  int m, int n, int lnz, float _max, int denwin, int* ndep,
  __global float* diag,
  __global float* perm,
  __global int* iperm,
  __global float* At,
  __global int* iAt,
  __global int* kAt,
  __global float* A,
  __global int* iA,
  __global int* kA,
  __global float* AAt,
  __global int* iAAt,
  __global int* kAAt,
  //__global float* Q,
  //__global int* iQ,
  //__global int* kQ,
  __global float* dn,
  __global float* dm,
  __global float* fwork,
  __global int* iwork,
  __global int* mark
  ) {


  int i, j, k, m2;
  int row, col;
  float epsdiag = _EPSDIAG;
  m2 = m+n;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);
  //int wgid = get_group_id(0);

  /*----------------------------------------------+
  | Store the diagonal of K in diag[].            |
  |                                              */

  for (j=0; j<n; j++) {
    diag[gid+iperm[j]*gsize]   = -fmax(dn[gid+j*gsize],epsdiag);
    iwork[gid+j*gsize] = 0;
  }
  for (i=0; i<m; i++) {
    diag[gid+iperm[n+i]*gsize] =  fmax(dm[gid+i*gsize],epsdiag);
    iwork[gid+(n+i)*gsize] = 0;
  }
  /*----------------------------------------------+
  | Store lower triangle of permutation of K      |
  | in AAt[], iAAt[], kAAt[].                     |
  |                                              */

  for (j=0; j<n; j++) {
          col = iperm[j];                 /* col is a new_index */
          for (k=kAAt[col]; k<kAAt[col+1]; k++) {
                  iwork[gid+iAAt[k]*gsize] = k;
                  AAt[gid+k*gsize] = 0.0;
          }

          for (k=kA[j]; k<kA[j+1]; k++) {
                  row = iperm[n+iA[k]];   /* row is a new_index */
                  if (row > col) {
                    AAt[gid+iwork[gid+row*gsize]*gsize] = A[k];
                  }
          }
          /*
          for (k=kQ[j]; k<kQ[j+1]; k++) {
                  row = iperm[iQ[k]];     /* row is a new_index
                  if (row > col) AAt[iwork[row]] = -_max*Q[k];
                  else if (row == col) diag[row] -= _max*Q[k];
          }
          */
  }

  for (i=0; i<m; i++) {
          col = iperm[n+i];
          for (k=kAAt[col]; k<kAAt[col+1]; k++) {
            iwork[gid+iAAt[k]*gsize] = k;
            AAt[gid+k*gsize] = 0.0;
          }

          for (k=kAt[i]; k<kAt[i+1]; k++) {
                  row = iperm[iA[k]];
                  if (row > col) {
                    AAt[gid+iwork[gid+row*gsize]*gsize] = At[k];
                  }
          }
  }


  /*----------------------------------------------+
  | Going into lltnum, any row/column for which   |
  | mark[] is set to FALSE will be treated as     |
  | a non-existing row/column.  On return,        |
  | any dependent rows are also marked FALSE.     |
  |                                              */

  for (i=gid; i<gsize*(m+n); i+=gsize) mark[i] = TRUE;

  lltnum(m, n, lnz, _max, denwin, ndep, diag, perm, iperm,
    A, iA, kA, At, iAt, kAt, AAt, iAAt, kAAt, //Q, iQ, kQ,
    dn, dm, fwork, iwork, mark
    );

}

/*----------------------------------------------+
| The routine rawsolve() does the forward,      |
| diagonal, and backward substitions to solve   |
| systems of equations involving the known      |
| factorization.                               */

void rawsolve(
	int m, int n, int lnz, int* ndep,
  __global float* diag,
  __global float* AAt,
  __global int* iAAt,
  __global int* kAAt,
  __global float* z,
  __global int* mark,
  int consistent
)
{
  int i;
  int m2, k, row;
  m2 = m+n;
  float eps = 0.0;
	float beta;
  float epssol = _EPSSOL;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);

  consistent = TRUE;


  if ((*ndep)) {
    maxv(z,m,&eps);
    eps *= epssol;
  }

  /*------------------------------------------------------+
  |                                                       |
  |               -1                                      |
  |       z  <-  L  z                                     |
  |                                                      */

  for (i=0; i<m2; i++) {
    if (mark[gid+i*gsize]) {
      beta = z[gid+i*gsize];
      for (k=kAAt[i]; k<kAAt[i+1]; k++) {
        row = iAAt[k];
        z[gid+row*gsize] -= AAt[gid+k*gsize]*beta;
      }
    } else if ( fabs(z[gid+i*gsize]) > eps ) {
      consistent = FALSE;
    } else {
      z[gid+i*gsize] = 0.0;
    }
  }

  /*------------------------------------------------------+
  |                                                       |
  |               -1                                      |
  |       z  <-  D  z                                     |
  |                                                      */

  for (i=m2-1; i>=0; i--) {
    if (mark[gid+i*gsize]) {
      z[gid+i*gsize] = z[gid+i*gsize]/diag[gid+i*gsize];
    } else if ( fabs(z[gid+i*gsize]) > eps ) {
      consistent = FALSE;
    } else {
      z[gid+i*gsize] = 0.0;
    }
  }

  /*------------------------------------------------------+
  |                                                       |
  |                t -1                                   |
  |       z  <-  (L )  z                                  |
  |                                                      */

  for (i=m2-1; i>=0; i--) {
    if (mark[gid+i*gsize]) {
      beta = z[gid+i*gsize];
      for (k=kAAt[i]; k<kAAt[i+1]; k++) {
        beta -= AAt[gid+k*gsize]*z[gid+iAAt[k]*gsize];
      }
      z[gid+i*gsize] = beta;
    } else if ( fabs(z[gid+i*gsize]) > eps ) {
      consistent = FALSE;
    } else {
      z[gid+i*gsize] = 0.0;
    }
  }

}



/*----------------------------------------------+
| The routine solve() uses rawsolve() together  |
| with iterative refinement to produce the best |
| possible solutions for the given              |
| factorization.                               */

void forwardbackward(
  int m, int n, int lnz, int _max, int* ndep,
  __global float* diag,
  __global int* iperm,
  __global float* At,
  __global int* iAt,
  __global int* kAt,
  __global float* A,
  __global int* iA,
  __global int* kA,
  __global float* AAt,
  __global int* iAAt,
  __global int* kAAt,
  //__global float* Q,
  //__global int* iQ,
  //__global int* kQ,
  __global int* mark,
  __global float* Dn,   /* diagonal matrix for upper-left  corner */
  __global float* Dm,    /* diagonal matrix for lower-right corner */
  __global float* c,
  __global float* b,
  __global float* fwork, // At least 3m+4n in length
  int consistent,
  int verbose
)
{
  int i, j, pass=0;
	int m2 = m+n;
	float maxrs, oldmaxrs, maxbc;
  float temp1, temp2;
  __global float *x_k, *y_k, *r, *s, *z;//, *Qx;
  int gid = get_global_id(0);
  int gsize = get_global_size(0);

  consistent = TRUE;

  y_k = fwork; // length m
  x_k = y_k+m*gsize; // length n
  r = x_k+n*gsize; // length m
  s = r+m*gsize; // length n
  z = s+n*gsize; // length n+m
  //Qx = z+m2; // length n


  maxv(b,m,&temp1);
  maxv(c,n,&temp2);

	maxbc = fmax( temp1, temp2 ) + 1.0f;

	maxrs = HUGE_VALF;
	do {
    if (pass == 0) {
              for (j=0; j<n; j++) z[gid+iperm[j]*gsize]   = c[gid+j*gsize];
              for (i=0; i<m; i++) z[gid+iperm[n+i]*gsize] = b[gid+i*gsize];
    } else {
              for (j=0; j<n; j++) z[gid+iperm[j]*gsize]   = s[gid+j*gsize];
              for (i=0; i<m; i++) z[gid+iperm[n+i]*gsize] = r[gid+i*gsize];
    }

	    rawsolve(m,n,lnz,ndep,diag,AAt,iAAt,kAAt,z,mark,consistent);

      if (pass == 0) {
                for (j=0; j<n; j++) x_k[gid+j*gsize] = z[gid+iperm[j]*gsize];
                for (i=0; i<m; i++) y_k[gid+i*gsize] = z[gid+iperm[n+i]*gsize];
	    } else {
                for (j=0; j<n; j++) x_k[gid+j*gsize] = x_k[gid+j*gsize] + z[gid+iperm[j]*gsize];
                for (i=0; i<m; i++) y_k[gid+i*gsize] = y_k[gid+i*gsize] + z[gid+iperm[n+i]*gsize];
	    }

	    smx(m,n,A, kA, iA, x_k,r);
	    smx(n,m,At,kAt,iAt,y_k,s);
	    //smx_ll(n,n,Q ,kQ ,iQ ,x_k,Qx);

	    for (j=gid; j<n*gsize; j+=gsize) {
		    //s[j] = c[j] - (s[j] - Dn[j]*x_k[j] - _max*Qx[j]);
        s[j] = c[j] - (s[j] - Dn[j]*x_k[j] );
	    }

	    for (i=gid; i<gsize*m; i+=gsize) {
		    r[i] = b[i] - (r[i] + Dm[i]*y_k[i]);
	    }

	    oldmaxrs = maxrs;
      maxv(r,m,&temp1);
      maxv(s,n,&temp2);
	    maxrs = fmax( temp1,temp2 );

	    /* --- for tuning purposes --- */
      #if __OPENCL_C_VERSION__ >= CL_VERSION_1_2
      if (verbose>0 && pass>0) {
        maxv(s,n,&temp1);
        maxv(r,m,&temp2);
		    printf("refinement(%3d): %8.2e %8.2e %8.2e \n",
		   pass, temp1, temp2, maxrs/maxbc );
	    }
      #endif



	    pass++;
	} while( maxrs > 1.0e-10*maxbc && maxrs < oldmaxrs/2 );

	if ( maxrs > oldmaxrs && pass > 1 ) {
    for (j=0; j<n; j++) x_k[gid+j*gsize] = x_k[gid+j*gsize] - z[gid+iperm[j]*gsize];
    for (i=0; i<m; i++) y_k[gid+i*gsize] = y_k[gid+i*gsize] - z[gid+iperm[n+i]*gsize];
	}

	/*----------------------------------------------------------
	| overwrite input with output                             */

  for (j=gid; j<n*gsize; j+=gsize) c[j] = x_k[j];
  for (i=gid; i<gsize*m; i+=gsize) b[i] = y_k[i];

}
