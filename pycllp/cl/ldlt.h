__kernel void inv_num(
  int m, int n, float _max, int denwin, int ndep,
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
  __global float* Q,
  __global int* iQ,
  __global int* kQ,
  __local float* dn,
  __local float* dm,
  __local float* fwork,
  __local int* iwork,
  __local int* mark
);
void lltnum(
  int m, int n, float _max,  int denwin, int ndep,
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
  __global float* Q,
  __global int* iQ,
  __global int* kQ,
  __local float* dn,
  __local float* dm,
  __local float* temp,
  __local int* iwork,
  __local int* mark
);
void forwardbackward(
  int m, int n, int _max, int ndep,
  __local float* diag,
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
  __global float* Q,
  __global int* iQ,
  __global int* kQ,
  __local int* mark,
  __local float* Dn,   /* diagonal matrix for upper-left  corner */
  __local float* Dm,    /* diagonal matrix for lower-right corner */
  __local float* c,
  __local float* b,
  __local float* fwork, // At least 3m+4n in length
  int consistent
);