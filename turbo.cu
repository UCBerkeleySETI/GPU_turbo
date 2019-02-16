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


// Neighbour pixel generator (N-W to W order).
//__constant__ int N_xs[8] = {0, -1, 1,-1,1,-1,0,1};
//__constant__ int N_ys[8] = {-1,-1,-1,,0,0, 1,1,1};

__global__ void threshold_and_local_max(float *g_idata, bool *g_odata, const float thresh, const int nfreqs, const int ndelays) {
    // 
    int tx = threadIdx.x;  int ty = threadIdx.y;
    int bx = blockIdx.x;   int by = blockIdx.y;
    int bdx = blockDim.x;  int bdy = blockDim.y;
    int i = bdx * bx + tx; int j = bdy * by + ty; 
    int p = INDEX(j,i,nfreqs); //j is delays, i is freqs

    if (i >= nfreqs || j >= ndelays) return;

    //Mirror boundary condition
    int i_left; int i_right; int j_down; int j_up;
    if(i==0) {i_left=i;} else {i_left=i-1;}   
    if(i==nfreqs-1) {i_right=i;} else {i_right=i+1;}
    if(j==0) {j_down=j;} else {j_down=j-1;}
    if(j==ndelays-1) {j_up=j;} else {j_up=j+1;}
    int n_neigh; 
    int* neighbors = NULL;

    switch (%(CONN)s ) {
          case 1:
            n_neigh = 4;
            int neigh1 [4] = {j*nfreqs+i_left, j_down*nfreqs+i, j*nfreqs+i_right, j_up*nfreqs+i};
            neighbors = neigh1;
            break;
          case 2:
            n_neigh = 8;
            int neigh2 [8] = {j*nfreqs+i_left, 
                            j_down*nfreqs+i_left, j_down*nfreqs+i, j_down*nfreqs+i_right,
                            j*nfreqs+i_right, j_up*nfreqs+i_right, j_up*nfreqs+i,
                            j_up*nfreqs+i_left};
            neighbors = neigh2;
            break;
    }


    if (g_idata[p] > thresh){
        g_odata[p] = true;
        int ne;
        float nei_max = g_idata[p];
        for (int ni=0; ni<n_neigh; ni++) 
        {
            ne = neighbors[ni];
            if ( g_idata[ne] > nei_max)               
            {
                g_odata[p] = false;
                break;
            }
        }
    }
    else {
        g_odata[p] = false;
    }
}