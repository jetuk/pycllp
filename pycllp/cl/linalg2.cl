
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
    int gid = get_global_id(0);
    int gsize = get_global_size(0);

    for (i=gid; i<n*gsize; i+=gsize) {
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
  int gid = get_global_id(0);
  int gsize = get_global_size(0);


  for (i=0; i<m; i++) y[gid+i*gsize] = 0.0e0;
  for (j=0; j<n; j++)
          for (k=ka[j]; k<ka[j+1]; k++)
                  y[gid+ia[k]*gsize] += a[k]*x[gid+j*gsize];

}


/*---------------------------------------------------------------+
|  compute componentwise maximum of n-vector x                  */

void maxv( __global float *x, int n, float *maxv)
{
        int i;
        int gid = get_global_id(0);
        int gsize = get_global_size(0);
        (*maxv)=0.0e0;
        for (i=gid; i<n*gsize; i+=gsize) (*maxv) = fmax((*maxv), fabs(x[i]));

}
