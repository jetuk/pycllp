
void fill(
  __global float* x,
  float val
  ) {

    int wgid = get_group_id(0);
    x[wgid] = val;
  }

void dotprod(
  __global float* x,
  __global float* y,
  int n,
  float *res
  ) {
    int i;
    (*res) = 0.0;
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int wgid = get_group_id(0);

    for (i=0; i<n; i++) {
      (*res) += x[wgid*n+i]*y[wgid*n+i];
    }

  }

void dotprod_gl(
  __global float* x,
  __local float* y,
  int n,
  float *res
  ) {
    int i;
    (*res) = 0.0;

    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int wgid = get_group_id(0);

    for (i=0; i<n; i++) {
      //printf("OCL: %d dotprod %5.1f %5.1f\n", wgid, x[wgid*n+i], y[i]);
      (*res) += x[wgid*n+i]*y[i];
    }
    //printf("%5.1f\n", res);

  }

void dotprod_ll(
  __local float* x,
  __local float* y,
  int n,
  float *res
  ) {
    int i;
    (*res) = 0.0;

    int lid = get_local_id(0);
    int gid = get_global_id(0);

    for (i=0; i<n; i++) {
      (*res) += x[i]*y[i];
    }

  }

void smx(
  int m, int n,
  __global float* a,
  __global int* ka,
  __global int* ia,
  __global float* x,
  __global float* y)
{
  int i,j,k;
  int lid = get_local_id(0);
  int gid = get_global_id(0);
  int wgid = get_group_id(0);


  for (i=0; i<m; i++) y[wgid*m+i] = 0.0e0;
  for (j=0; j<n; j++)
          for (k=ka[j]; k<ka[j+1]; k++)
                  y[wgid*m+ia[k]] += a[k]*x[wgid*n+j];

}

void smx_gl(
  int m, int n,
  __global float* a,
  __global int* ka,
  __global int* ia,
  __global float* x,
  __local float* y)
{
  int i,j,k;
  int lid = get_local_id(0);
  int gid = get_global_id(0);

  for (i=0; i<m; i++) y[i] = 0.0e0;
  for (j=0; j<n; j++)
          for (k=ka[j]; k<ka[j+1]; k++)
                  y[ia[k]] += a[k]*x[gid*n+j];

}

void smx_ll(
  int m, int n,
  __global float* a,
  __global int* ka,
  __global int* ia,
  __local float* x,
  __local float* y)
{
  int i,j,k;

  for (i=0; i<m; i++) y[i] = 0.0e0;
  for (j=0; j<n; j++)
          for (k=ka[j]; k<ka[j+1]; k++) {
                  y[ia[k]] += a[k]*x[j];
          }

}


/*---------------------------------------------------------------+
|  compute componentwise maximum of n-vector x                  */

void maxv_l( __local float *x, int n, float *maxv)
{
        int i;
        (*maxv)=0.0e0;
        for (i=0; i<n; i++) (*maxv) = max((*maxv), fabs(x[i]));

}
