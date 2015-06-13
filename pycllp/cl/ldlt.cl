#include <linalg.h>


#define	TRUE 1
#define	FALSE 0


#define _EPS 1.0e-6
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
  int m, int n, float _max,  int denwin, int* ndep,
  __local float* diag,
  __global float* perm,
  __global int* iperm,
  __global float* A,
  __global int* iA,
  __global int* kA,
  __global float* At,
  __global int* iAt,
  __global int* kAt,
  __local float* AAt,
  __global int* iAAt,
  __global int* kAAt,
  //__global float* Q,
  //__global int* iQ,
  //__global int* kQ,
  __local float* dn,
  __local float* dm,
  __local float* temp,
  __local int* iwork,
  __local int* mark
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
  // local iwork memory pointers
  __local int *first = iwork;
  __local int *link = iwork+m2;
  int wgid = get_group_id(0);

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

        for (i=0; i<m2; i++) {
          temp[i] = 0.0;
          first[i] = 0;
          link[i] = -1;
        }

        for (i=0; i<m2; i++) {
                if (fabs(diag[i]) > maxdiag) maxdiag = fabs(diag[i]);
        }

        (*ndep)=0;

        /*------------------------------------------------------+
        | begin main loop - this code is taken from George and  |
        | Liu's book, pg. 155, modified to do LDLt instead      |
        | of LLt factorization.                                */

        for (i=0; i<m2; i++) {
          diagi = diag[i];
	        sgn_diagi = perm[i] < n ? -1 : 1;
            for (j=link[i]; j != -1; j=newj) {
                newj = link[j];
                k = first[j];
                lij = AAt[k];
                lij_dj = lij*diag[j];
                diagi -= lij*lij_dj;
                k_bgn = k+1;
                k_end = kAAt[j+1];
                if (k_bgn < k_end) {
                    first[j] = k_bgn;
                    row = iAAt[k_bgn];
                    link[j] = link[row];
                    link[row] = j;
                    if (j < denwin) {
                        for (kk=k_bgn; kk<k_end; kk++)
                            temp[iAAt[kk]] += lij_dj*AAt[kk];
                    } else {
                        int ptr;
                        ptr = row;
                        for (kk=k_bgn; kk<k_end; kk++) {
                            temp[ptr] += lij_dj*AAt[kk];
                            ptr++;
                        }
                    }
                }
            }
            k_bgn = kAAt[i];
            k_end = kAAt[i+1];
            for (kk=k_bgn; kk<k_end; kk++) {
                row = iAAt[kk];
		            AAt[kk] -= temp[row];
            }
            if (fabs(diagi) <= epsnum*maxdiag || mark[i] == FALSE) {
	    /*
            if (sgn_diagi*diagi <= epsnum*maxdiag || mark[i] == FALSE)
	    */
                (*ndep)++;
		maxoffdiag = 0.0;
                for (kk=k_bgn; kk<k_end; kk++) {
		    maxoffdiag = fmax( maxoffdiag, fabs( AAt[kk] ) );
                }
		if ( maxoffdiag < 1.0e+6*eps ) {
		    mark[i] = FALSE;
		} else {
		    diagi = sgn_diagi * eps;
		}
            }
	    diag[i] = diagi;
            if (k_bgn < k_end) {
                first[i] = k_bgn;
                row = iAAt[k_bgn];
                link[i] = link[row];
                link[row] = i;
                for (kk=k_bgn; kk<k_end; kk++) {
                    row = iAAt[kk];
                    if (mark[i]) {
                            AAt[kk] /= diagi;
                    } else {
                            AAt[kk] = 0.0;
		    }
                    temp[row] = 0.0;
                }
            }
        }

}

void inv_num(
  int m, int n, float _max, int denwin, int* ndep,
  __local float* diag,
  __global float* perm,
  __global int* iperm,
  __global float* At,
  __global int* iAt,
  __global int* kAt,
  __global float* A,
  __global int* iA,
  __global int* kA,
  __local float* AAt,
  __global int* iAAt,
  __global int* kAAt,
  //__global float* Q,
  //__global int* iQ,
  //__global int* kQ,
  __local float* dn,
  __local float* dm,
  __local float* fwork,
  __local int* iwork,
  __local int* mark
  ) {


  int i, j, k;
  int row, col;
  float epsdiag = _EPSDIAG;
  int wgid = get_group_id(0);

  /*----------------------------------------------+
  | Store the diagonal of K in diag[].            |
  |                                              */

  for (j=0; j<n; j++) { diag[iperm[j]]   = -fmax(dn[j],epsdiag); }
  for (i=0; i<m; i++) { diag[iperm[n+i]] =  fmax(dm[i],epsdiag); }

  /*----------------------------------------------+
  | Store lower triangle of permutation of K      |
  | in AAt[], iAAt[], kAAt[].                     |
  |                                              */

  for (j=0; j<n; j++) {
          col = iperm[j];                 /* col is a new_index */
          for (k=kAAt[col]; k<kAAt[col+1]; k++) {
                  iwork[iAAt[k]] = k;
                  AAt[k] = 0.0;
          }

          for (k=kA[j]; k<kA[j+1]; k++) {
                  row = iperm[n+iA[k]];   /* row is a new_index */
                  if (row > col) {
                    AAt[iwork[row]] = A[k];
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
                  iwork[iAAt[k]] = k;
                  AAt[k] = 0.0;

          }

          for (k=kAt[i]; k<kAt[i+1]; k++) {
                  row = iperm[iAt[k]];
                  if (row > col) {
                    AAt[iwork[row]] = At[k];
                  }
          }
  }


  /*----------------------------------------------+
  | Going into lltnum, any row/column for which   |
  | mark[] is set to FALSE will be treated as     |
  | a non-existing row/column.  On return,        |
  | any dependent rows are also marked FALSE.     |
  |                                              */

  for (i=0; i<m+n; i++) mark[i] = TRUE;

  lltnum(m, n, _max, denwin, ndep, diag, perm, iperm,
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
	int m, int n, int* ndep,
  __local float* diag,
  __local float* AAt,
  __global int* iAAt,
  __global int* kAAt,
	__local float* z,
  __local int* mark,
  int consistent
)
{
  int i;
  int m2, k, row;
  float eps = 0.0;
	float beta;
  float epssol = _EPSSOL;
  consistent = TRUE;
  m2 = m+n;

  if ((*ndep)) {
    maxv_l(z,m,&eps);
    eps *= epssol;
  }

  /*------------------------------------------------------+
  |                                                       |
  |               -1                                      |
  |       z  <-  L  z                                     |
  |                                                      */

  for (i=0; i<m2; i++) {
    if (mark[i]) {
      beta = z[i];
      for (k=kAAt[i]; k<kAAt[i+1]; k++) {
        row = iAAt[k];
        z[row] -= AAt[k]*beta;
      }
    } else if ( fabs(z[i]) > eps ) {
      consistent = FALSE;
    } else {
      z[i] = 0.0;
    }
  }

  /*------------------------------------------------------+
  |                                                       |
  |               -1                                      |
  |       z  <-  D  z                                     |
  |                                                      */

  for (i=m2-1; i>=0; i--) {
    if (mark[i]) {
      z[i] = z[i]/diag[i];
    } else if ( fabs(z[i]) > eps ) {
      consistent = FALSE;
    } else {
      z[i] = 0.0;
    }
  }

  /*------------------------------------------------------+
  |                                                       |
  |                t -1                                   |
  |       z  <-  (L )  z                                  |
  |                                                      */

  for (i=m2-1; i>=0; i--) {
    if (mark[i]) {
      beta = z[i];
      for (k=kAAt[i]; k<kAAt[i+1]; k++) {
        beta -= AAt[k]*z[iAAt[k]];
      }
      z[i] = beta;
    } else if ( fabs(z[i]) > eps ) {
      consistent = FALSE;
    } else {
      z[i] = 0.0;
    }
  }

}



/*----------------------------------------------+
| The routine solve() uses rawsolve() together  |
| with iterative refinement to produce the best |
| possible solutions for the given              |
| factorization.                               */

void forwardbackward(
  int m, int n, int _max, int* ndep,
  __local float* diag,
  __global int* iperm,
  __global float* At,
  __global int* iAt,
  __global int* kAt,
  __global float* A,
  __global int* iA,
  __global int* kA,
  __local float* AAt,
  __global int* iAAt,
  __global int* kAAt,
  //__global float* Q,
  //__global int* iQ,
  //__global int* kQ,
  __local int* mark,
  __local float* Dn,   /* diagonal matrix for upper-left  corner */
  __local float* Dm,    /* diagonal matrix for lower-right corner */
  __local float* c,
  __local float* b,
  __local float* fwork, // At least 3m+4n in length
  int consistent,
  int verbose
)
{
  int i, j, pass=0;
	int m2 = m+n;
	float maxrs, oldmaxrs, maxbc;
  float temp1, temp2;
  __local float *x_k, *y_k, *r, *s, *z;//, *Qx;
  consistent = TRUE;

  y_k = fwork; // length m
  x_k = y_k+m; // length n
  r = x_k+n; // length m
  s = r+m; // length n
  z = s+n; // length n+m
  //Qx = z+m2; // length n


  maxv_l(b,m,&temp1);
  maxv_l(c,n,&temp2);

	maxbc = fmax( temp1, temp2 ) + 1.0f;

	maxrs = HUGE_VALF;
	do {
	    if (pass == 0) {
                for (j=0; j<n; j++) {
                  z[iperm[j]]   = c[j];
                }
                for (i=0; i<m; i++) {
                  z[iperm[n+i]] = b[i];
                }
	    } else {
                for (j=0; j<n; j++) z[iperm[j]]   = s[j];
                for (i=0; i<m; i++) z[iperm[n+i]] = r[i];
	    }

	    rawsolve(m,n,ndep,diag,AAt,iAAt,kAAt,z,mark,consistent);

	    if (pass == 0) {
                for (j=0; j<n; j++) {
                  x_k[j] = z[iperm[j]];
                }
                for (i=0; i<m; i++) {
                  y_k[i] = z[iperm[n+i]];
                }
	    } else {
                for (j=0; j<n; j++) x_k[j] = x_k[j] + z[iperm[j]];
                for (i=0; i<m; i++) y_k[i] = y_k[i] + z[iperm[n+i]];
	    }

	    smx_ll(m,n,A, kA, iA, x_k,r);
	    smx_ll(n,m,At,kAt,iAt,y_k,s);
	    //smx_ll(n,n,Q ,kQ ,iQ ,x_k,Qx);

	    for (j=0; j<n; j++) {
		    //s[j] = c[j] - (s[j] - Dn[j]*x_k[j] - _max*Qx[j]);
        s[j] = c[j] - (s[j] - Dn[j]*x_k[j] );
	    }

	    for (i=0; i<m; i++) {
		    r[i] = b[i] - (r[i] + Dm[i]*y_k[i]);
	    }

	    oldmaxrs = maxrs;
      maxv_l(r,m,&temp1);
      maxv_l(s,n,&temp2);
	    maxrs = fmax( temp1,temp2 );

	    /* --- for tuning purposes --- */
      #if __OPENCL_C_VERSION__ >= CL_VERSION_1_2
      if (verbose>0 && pass>0) {
        maxv_l(s,n,&temp1);
        maxv_l(r,m,&temp2);
		    printf("refinement(%3d): %8.2e %8.2e %8.2e \n",
		   pass, temp1, temp2, maxrs/maxbc );
	    }
      #endif



	    pass++;
	} while( maxrs > 1.0e-10*maxbc && maxrs < oldmaxrs/2 );

	if ( maxrs > oldmaxrs && pass > 1 ) {
            for (j=0; j<n; j++) x_k[j] = x_k[j] - z[iperm[j]];
            for (i=0; i<m; i++) y_k[i] = y_k[i] - z[iperm[n+i]];
	}

	/*----------------------------------------------------------
	| overwrite input with output                             */

  for (j=0; j<n; j++) c[j] = x_k[j];
  for (i=0; i<m; i++) b[i] = y_k[i];

}
