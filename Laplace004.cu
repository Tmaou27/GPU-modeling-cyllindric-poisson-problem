#include <stdio.h>
#include <ctime>

//parameters of model
__constant__ int k = 10;//scale, k cells in 1 cm

//sizes in cm
__constant__ int r = 1 ;//Radius of ball
__constant__ int d_1 = 8;//Distance between ball`s center and right plate
__constant__ int d_2 = 4;//===========same====================left plate
__constant__ int a = 20;//Diameter of plate
__constant__ int a_thikness = 1;//Thickness of plate

//voltage
__constant__ int phi_0 = 10000;//Absolute value of potential of plates
__constant__ int phi_1 = 10000;//Potential of ball


//variables
__device__ float q = 0;


//functions
__device__ float sq(float a)
{
  return a*a;
}
__device__ int sq(int a)
{
  return a*a;
}
__device__ int sign(int i)
{
    if(i == 0)
    {
      return 0;
    }
    if( i > 0)
    {
      return 1;
    }
    return -1;
}
__global__ void exchange(int n, float* d_border_conditions, float* d_mapa, float* d_new_mapa)
{
  int x = blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;
  int y = blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;
  int localX = threadIdx.x;
  int localY = threadIdx.y;

  int i = x + n*y;
  if(x<0||x>=n||y<0||y>=n){return;}
  bool isGalo = (localX==0)||(localX==blockDim.x-1)||(localY==0)||(localY==blockDim.y-1);
  if(isGalo){
    return;
  }
  d_mapa[i] = d_new_mapa[i];
}

__global__ void initial(int n, float* d_border_conditions, float* d_mapa, float* d_new_mapa)
{
  int x = blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;
  int y = blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;
  int localX = threadIdx.x;
  int localY = threadIdx.y;

  int i = x + n*y;
  if(x<0||x>=n||y<0||y>=n){return;}
  bool isGalo = (localX==0)||(localX==blockDim.x-1)||(localY==0)||(localY==blockDim.y-1);
  if(isGalo){
    return;
  }

  int p[] = {x, y};
  if( sq(x - n/2) + sq(y - n/2) <= sq(k*r) )
  {
      //ball
      d_mapa[i] = phi_1;
      d_new_mapa[i] = phi_1;
      d_border_conditions[i] = 1;
      return;
  }
  if( p[1] >= n/2-k*a && p[1] <= n/2+k*a && p[0] >= n/2+k*d_1-a_thikness && p[0] <= n/2+k*d_1+a_thikness )
  {
      //plate +
      d_mapa[i] = phi_0;
      d_new_mapa[i] = phi_0;
      d_border_conditions[i] = 1;
      return;
  }
  if( p[1] >= n/2-k*a && p[1] <= n/2+k*a && p[0] >= n/2-k*d_2-a_thikness && p[0] <= n/2-k*d_2+a_thikness )
  {
      //plate -
      d_mapa[i] = -phi_0;
      d_new_mapa[i] = -phi_0;
      d_border_conditions[i] = 1;
      return;
  }
  if( p[0] == 0 || p[0] == n-1 || p[1] == 0 || p[1] == n - 1 )
  {
      //border
      d_mapa[i] = 0;
      d_new_mapa[i] = 0;
      d_border_conditions[i] = 1;
      return;
  }
  d_mapa[i] = 0;
  d_new_mapa[i] = 0;
  d_border_conditions[i] = 0;
}
__global__ void calculateCharge(int n, float* d_border_conditions, float* d_mapa, float* d_new_mapa)
{
  __shared__ float l_mapa[32][32];

  int x = blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;
  int localX = threadIdx.x;
  int y = blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;
  int localY = threadIdx.y;

  int i = x + n*y;
  l_mapa[localX][localY] = (x >= 0 && x < n && y >= 0 && y < n) ? d_mapa[i] : 0.0f;
  __syncthreads();
  bool isGalo = (localX==0)||(localX==blockDim.x-1)||(localY==0)||(localY==blockDim.y-1);

  if((x<n/2+r*k + 2)&&(x>=n/2-r*k - 3)&&(y<n/2 +r*k + 2)&&(y>=n/2)&&!isGalo)
  {
        //printf("%g\n", l_mapa[localX][localY]);
        atomicAdd(&q, l_mapa[localX][localY+ 1] - l_mapa[localX][localY]);
        atomicAdd(&q, abs(y-n/2) *
        (-4 * l_mapa[localX][localY]
          + l_mapa[localX+1][localY] + l_mapa[localX-1][localY]
          + l_mapa[localX][localY-1] + l_mapa[localX][localY+1]));
  }
}
__global__ void compute(int n, float* d_border_conditions, float* d_mapa, float* d_new_mapa, bool show)
{
  __shared__ float l_mapa[32][32];

  int x = blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;
  int localX = threadIdx.x;
  int y = blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;
  int localY = threadIdx.y;

  int i = x + n*y;
  l_mapa[localX][localY] = (x >= 0 && x < n && y >= 0 && y < n) ? d_mapa[i] : 0.0f;
  if(show)
  {
    if(1||l_mapa[localX][localY]!= 0)
    {
      printf("%g ", l_mapa[localX][localY]);
    }
  }
  __syncthreads();

  bool isGalo = (localX==0)||(localX==blockDim.x-1)||(localY==0)||(localY==blockDim.y-1);
  bool isBorder = d_border_conditions[i];

  int p[] = {x, y};
  //Jacobi iteration
  float delta = 0;
  if(p[1]-n/2 != 0 && !isBorder && !isGalo)
  {
    delta = 1/(float)(1/(float)abs(p[1]-n/2)+4) *
    (l_mapa[localX-1][localY]
    +(1/(float)abs(p[1]-n/2)+1) *l_mapa[localX][localY+sign(p[1]-n/2)]
    +l_mapa[localX+1][localY]
    +l_mapa[localX][localY-sign(p[1]-n/2)]);

  }
  if(p[1]-n/2 == 0 && !isBorder && !isGalo)
  {
    delta = 1/(float)6 * (2*(l_mapa[localX][localY-1]  + l_mapa[localX][localY+1]) + l_mapa[localX-1][localY] + l_mapa[localX+1][localY]);
  }
  if(!isBorder && !isGalo)
  {
    d_new_mapa[i] = delta;
  }
  if(isBorder && !isGalo)
  {
    d_new_mapa[i] = l_mapa[localX][localY];
  }
}

int main()
{
    clock_t start, stop;
    start = clock();

    float* d_border_conditions;
    float* d_mapa;
    float* d_new_mapa;

    int h_n = 1909;

    dim3 threads(32, 32);
    dim3 blocks((h_n + (threads.x -2)- 1)/(threads.x -2),
                 (h_n + (threads.y -2) - 1)/(threads.y -2));
    dim3 blocksNoGalo((h_n + threads.x -1)/threads.x, (h_n + threads.y - 1)/threads.y);

    cudaMalloc(&d_border_conditions, (h_n*h_n) * sizeof(float));
    cudaMalloc(&d_mapa, (h_n*h_n) * sizeof(float));
    cudaMalloc(&d_new_mapa, (h_n*h_n) * sizeof(float));

    initial<<<blocks, threads>>>(h_n, d_border_conditions, d_mapa, d_new_mapa);
    cudaDeviceSynchronize();
    printf("Initial conditions done %d %d\n", blocks.x, blocks.y);


    for(int t = 0; t < 1000; ++t)
    {
      compute<<<blocks, threads>>>(h_n, d_border_conditions, d_mapa, d_new_mapa,0);
      cudaDeviceSynchronize();

      float* temp = d_mapa;
      d_mapa = d_new_mapa;
      d_new_mapa = temp;

      cudaError_t err = cudaPeekAtLastError();
      if (err != cudaSuccess)
      {
        // Handle the error
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
      }
    }
    printf("Iters done\n");

    calculateCharge<<<blocks, threads>>>(h_n, d_border_conditions, d_mapa, d_new_mapa);
    cudaDeviceSynchronize();
    float host_q;
    cudaMemcpyFromSymbol(&host_q, q, sizeof(float), 0, cudaMemcpyDeviceToHost);
    //Charge is proportional the length of cell size so we  divide by it
    int host_k;
    cudaMemcpyFromSymbol(&host_k, k, sizeof(int), 0, cudaMemcpyDeviceToHost);
    host_q *= -8.85 * pow(10, -12) * 3.1415 * 2 / (float)(host_k * 100);
    printf("%g\n", host_q);
    printf("Done\n");

    cudaDeviceSynchronize();
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        // Handle the error
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_border_conditions);
    cudaFree(d_mapa);
    cudaFree(d_new_mapa);
    //printf("Done\n");


    stop = clock();
    printf("t = %g\n", (double)(stop-start)/ CLOCKS_PER_SEC);
    return 0;
}