void inv_num(
  int m, int n, int lnz, float _max, int denwin, int* ndep,
  __global float* gdiag,
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
  __global float* fwork,
  __global int* iwork,
  __global int* mark
);
void lltnum(
  int m, int n, int lnz, float _max,  int denwin, int* ndep,
  __global float* gdiag,
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
);
void forwardbackward(
  int m, int n, int lnz, int _max, int* ndep,
  __global float* gdiag,
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
  __global int* mark,
  __global float* Dn,   /* diagonal matrix for upper-left  corner */
  __global float* Dm,    /* diagonal matrix for lower-right corner */
  __global float* c,
  __global float* b,
  __global float* fwork, // At least 3m+4n in length
  int consistent,
  int verbose
);
