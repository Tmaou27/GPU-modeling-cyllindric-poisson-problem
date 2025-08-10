#include <stdio.h>
#include <ctime>

//parameters of model
__constant__ int k = 4;//scale, k cells in 1 cm

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

__global__ void compute(int d_threads, int n, float* d_border_conditions, float* d_mapa, float* d_new_mapa)
{
  int tid = threadIdx.x;

  int c = (n*n)/d_threads+sign(n*n%d_threads);
  for(int i = c*tid; i < min(c*(tid+1), n*n); ++i)
  {
        int p[] = {i%n, i/n};
        if( sq(p[0] - n/2) + sq(p[1] - n/2) <= sq(k*r) )
        {
            //ball
            d_mapa[i] = phi_1;
            d_new_mapa[i] = phi_1;
            d_border_conditions[i] = 1;
            continue;
        }
        if( p[1] >= n/2-k*a && p[1] <= n/2+k*a && p[0] >= n/2+k*d_1-a_thikness && p[0] <= n/2+k*d_1+a_thikness )
        {
            //plate +
            d_mapa[i] = phi_0;
            d_new_mapa[i] = phi_0;
            d_border_conditions[i] = 1;
            continue;
        }
        if( p[1] >= n/2-k*a && p[1] <= n/2+k*a && p[0] >= n/2-k*d_2-a_thikness && p[0] <= n/2-k*d_2+a_thikness )
        {
            //plate -
            d_mapa[i] = -phi_0;
            d_new_mapa[i] = -phi_0;
            d_border_conditions[i] = 1;
            continue;
        }
        if( p[0] == 0 || p[0] == n-1 || p[1] == 0 || p[1] == n - 1 )
        {
            //border
            d_mapa[i] = 0;
            d_new_mapa[i] = 0;
            d_border_conditions[i] = 1;
            continue;
        }
    }
  __syncthreads();

  if(tid == 0)
  {
      //printf("Initial conditions done\n");
  }

  //Jacobi iteration method
  for(int t = 0; t < 3000; ++t)
  {
    for(int i = c*tid; i < min(c*(tid+1), n*n); ++i)
    {
        if(d_border_conditions[i] == 1){ continue; }
        int p[] = {i%n, i/n};
        float x = 0;
        if(p[1]-n/2 != 0)
        {
          x = 1/(float)(1/(float)abs(p[1]-n/2)+4) * (d_mapa[i-1]+(1/(float)abs(p[1]-n/2)+1)*d_mapa[i+sign(p[1]-n/2)*n]+d_mapa[i+1]+d_mapa[i-sign(p[1]-n/2)*n]);
        }
        else
        {
          x = 1/(float)6 * (2*(d_mapa[i-n] + d_mapa[i+n]) + d_mapa[i-1] + d_mapa[i+1]);
        }
        d_new_mapa[i] = x;
    }
    __syncthreads();
    for(int i = c*tid; i < min(c*(tid+1), n*n); ++i)
    {
        d_mapa[i] = d_new_mapa[i];
        //if(t == 2999)
        //{
         //   printf("%g\n", d_mapa[i]);
        //}

    }
    __syncthreads();
  }
  if(tid == 0)
  {
      //printf("Iterations done\n");
  }



  //Calculating charge
  if(tid == 0)
  {
      for(int i = n/2-r*k - 3; i < n/2+r*k + 2; ++i)
  {
    for(int j = n/2; j < n/2 +r*k  + 2; ++j)
    {
        atomicAdd(&q, d_mapa[i+ n*(j+1)] - d_mapa[i+ n*j]);
        atomicAdd(&q, abs(j-n/2) * (-4 * d_mapa[i+ n*j] + d_mapa[i+1+ n*j] +  d_mapa[i-1+ n*j] + d_mapa[i+ n*(j-1)] + d_mapa[i+ n*(j+1)]));
    }
  }
  }

  __syncthreads();
  //Charge is proportional the length of cell size so we  divide by it
  if(tid==0)
  {

    q = q * -8.85 * pow(10, -12) * 3.1415 * 2 / (float)(k * 100);
    //printf("%g\n", q);
    //printf("Charge calculated\n");
  }
  return;
}

int main()
{
  for(int i = 220; i < 550; i += 2){
    clock_t start, stop;
    start = clock();

    float* d_border_conditions;
    float* d_mapa;
    float* d_new_mapa;

    int threads = i;
    int h_n = 400;

    cudaMalloc(&d_border_conditions, (h_n*h_n) * sizeof(float));
    cudaMalloc(&d_mapa, (h_n*h_n) * sizeof(float));
    cudaMalloc(&d_new_mapa, (h_n*h_n) * sizeof(float));

    compute<<<1, threads>>>(threads, h_n, d_border_conditions, d_mapa, d_new_mapa);
    cudaDeviceSynchronize();
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        // Handle the error
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
    }

    float host_q;
    cudaMemcpyFromSymbol(&host_q, q, sizeof(float), 0, cudaMemcpyDeviceToHost);
    //printf("%g\n", host_q);


    cudaFree(d_border_conditions);
    cudaFree(d_mapa);
    cudaFree(d_new_mapa);
    //printf("Done\n");


    stop = clock();
    //printf("time = %g", (double)(stop-start)/ CLOCKS_PER_SEC);
    printf("%g\n", (double)(stop-start)/ CLOCKS_PER_SEC);
  }
    return 0;
}
