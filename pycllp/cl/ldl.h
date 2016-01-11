#ifndef real
  #define real double
#endif

__kernel void solve_primal_normal(int m, int n, __global real* A,
  __global double* x, __global double* z, __global double* y,
  __global real* b, __global real* c, real mu,
  __global real* L, __global real* D, __global double* S, __global double* dy,
  real delta);

__kernel void sparse_solve_primal_normal(int m, int n, __global real* A,
  __global double* x, __global double* z, __global double* y,
  __global real* b, __global real* c, real mu,
  __global real* Ldata, __constant int* Lindptr, __constant int* Lindices,
  __constant int* LTindptr, __constant int* LTindices, __constant int* LTmap,
  __global real* D, __global double* S, __global double* dy,
  real delta);