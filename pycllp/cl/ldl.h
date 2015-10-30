#ifndef real
  #define real double
#endif

__kernel void solve_primal_normal(int m, int n, __global real* A,
  __global real* x, __global real* z, __global real* y, __global real* w,
  __global real* b, __global real* c, real mu,
  __global real* L, __global real* D, __global real* dy);
