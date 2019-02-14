#define INDEX(j,i,ld) ((j) * ld + (i))
#define BDX %(BDX)s
#define BDY %(BDY)s
__device__ __constant__ float delay_table_const[%(NTIMES)s*%(NDELAYS)s];

__global__ void reduce(float *g_idata, float *g_odata, const int w) {

    // kernel to sum over y axis
    int tx = threadIdx.x;  int ty = threadIdx.y;
    int bx = blockIdx.x;   int by = blockIdx.y;
    int bdx = blockDim.x;  int bdy = blockDim.y;
    int i = bdx * bx + tx; int j = bdy * by + ty;
    int p = INDEX(j,i,w);
    __shared__ float sdata[BDX*BDY];

    sdata[INDEX(ty,tx,%(BDX)s)] = g_idata[p];

    __syncthreads();
    // do reduction in shared mem
    if (ty == 0) {
        for ( int t=1; t<bdy; t++) {
            sdata[INDEX(0,tx,BDX)] += sdata[INDEX(t,tx,BDX)];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (ty == 0)
        g_odata[i] = sdata[INDEX(0,tx,BDX)];
}

__global__ void sweep_const_mem(float *g_idata, float *g_odata, const int nfreqs, const int ntimes, const int ndelays) {

    // each thread loads one element from global to shared mem
    // note use of 1D thread indices (only) in this kernel
    int tx = threadIdx.x;  int ty = threadIdx.y;
    int bx = blockIdx.x;   int by = blockIdx.y;
    int bdx = blockDim.x;  int bdy = blockDim.y;
    int i = bdx * bx + tx; int j = bdy * by + ty;
    int p = INDEX(j,i,nfreqs);

    

    int delay;

    __syncthreads();
    // do reduction in shared mem
    for ( int t=0; t<ntimes; t++) {
        delay = delay_table_const[INDEX(t,j,ndelays)];
        if (delay+i >= 0 && delay+i < nfreqs){
            g_odata[p] += g_idata[t*nfreqs + i + delay];
        }
    }
}

__global__ void sweep(float *g_idata, float *g_odata, const int *delay_table, const int nfreqs, const int ntimes, const int ndelays) {
    // kernel sweeps over set of delays. This is the main kernel
    // each thread loads one element from global to shared mem
    // note use of 1D thread indices (only) in this kernel
    int tx = threadIdx.x;  int ty = threadIdx.y;
    int bx = blockIdx.x;   int by = blockIdx.y;
    int bdx = blockDim.x;  int bdy = blockDim.y;
    int i = bdx * bx + tx; int j = bdy * by + ty; 
    int p = INDEX(j,i,nfreqs); //j is delays, i is freqs

    int delay;

    __syncthreads();
    // each core computes one output pixel 
    for ( int t=0; t<ntimes; t++) {
        delay = delay_table[INDEX(t,j,ndelays)];
        if (delay+i >= 0 && delay+i < nfreqs){
            g_odata[p] += g_idata[t*nfreqs + i + delay];
        }
    }
}