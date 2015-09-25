void dotprod(
  __global float* x,
  __global float* y,
  int n,
  float *res
);

void smx(
  int m, int n,
  __global float* a,
  __global int* ka,
  __global int* ia,
  __global float* x,
  __global float* y);

void maxv( __global float *x, int n, float *maxv);
