void dotprod(
  __global float* x,
  __global float* y,
  int n,
  float *res
);
void dotprod_gl(
  __global float* x,
  __local float* y,
  int n,
  float *res
);
void dotprod_ll(
  __local float* x,
  __local float* y,
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
void smx_gl(
  int m, int n,
  __global float* a,
  __global int* ka,
  __global int* ia,
  __global float* x,
  __local float* y);


void maxv_l( __local float *x, int n, float *maxv);
