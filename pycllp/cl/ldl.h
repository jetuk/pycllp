#ifndef real
  #define real double
#endif

__kernel void solve_primal_normal(int m, int n, __global real* A,
  __global double* x, __global double* z, __global double* y,
  __global real* b, __global real* c, real mu,
  __global real* L, __global real* D, __global double* S, __global double* dy,
  real delta);
