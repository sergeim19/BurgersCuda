/* 
  Sergei Melkoumian
  April 27, 2015
  Burgers equation - GPU CUDA version
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cufft.h>
#include <time.h>
#include <sys/time.h>

#define NADVANCE (4000)
#define nu (5.0e-2)

int timeval_subtract (double *result, struct timeval *x, struct timeval *y)
{
  struct timeval result0;
    
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result0.tv_sec = x->tv_sec - y->tv_sec;
  result0.tv_usec = x->tv_usec - y->tv_usec;
  *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

__global__ void kernel_rescale_u(double *u_dev, int N)
{  
  int j;
  j = blockIdx.x * blockDim.x + threadIdx.x;

  u_dev[j] = u_dev[j] / (double)N;
}

__global__ void kernel_calc_uu(double *u_dev, double *uu_dev)
{  
  int j;
  j = blockIdx.x * blockDim.x + threadIdx.x;

  uu_dev[j] = 0.5 * u_dev[j] * u_dev[j];
}

__global__ void kernel_setmodezero(cufftDoubleComplex *fhat, int n)
{
  fhat[n].x = 0.0;
  fhat[n].y = 0.0;
}

__global__ void kernel_burgers(cufftDoubleComplex *uhat, cufftDoubleComplex *uuhat, double dt)
{  
  int jj;
  jj = blockIdx.x * blockDim.x + threadIdx.x;

  double j = (double)jj;

  uhat[jj].x = ( uhat[jj].x*(1.0 + j*j*nu*dt) + j*dt*uuhat[jj].y )/
    ( pow((1.0 + j*j*nu*dt), 2) );
  uhat[jj].y = ( uhat[jj].y*(1.0 + j*j*nu*dt) - j*dt*uuhat[jj].x )/
    ( pow((1.0 + j*j*nu*dt), 2) );

}

int main(void)
{
  FILE *out_file = fopen("data_burgers_gpu.dat", "w");
  int N = 1048576, blockSize = 512, nBlocks = N/blockSize;
  double dx = 2.0 * M_PI / (double)N, dt = 1.0e-3;
  double *x, *u, *u_dev, *uu_dev;
  double norm = 0.0;
  int devid, devcount, error;
  double restime;
  struct timeval  tdr0, tdr1;

  /*  find compute device an initialize it */
  /* add device detection */
  
  /* find number of device in current "context" */
  cudaGetDevice(&devid);
  /* find how many devices are available */
  if (cudaGetDeviceCount(&devcount) || devcount==0)
    {
      printf ("No CUDA devices!\n");
      exit (1);
    }
  else
    {
      cudaDeviceProp deviceProp; 
      cudaGetDeviceProperties (&deviceProp, devid);
      printf ("Device count, devid: %d %d\n", devcount, devid);
      printf ("Device: %s\n", deviceProp.name);
      printf("[deviceProp.major.deviceProp.minor] = [%d.%d]\n\n", deviceProp.major, deviceProp.minor);
    }

  x = (double *) malloc (sizeof (double) * N); 
  cudaMallocHost((void **) &u, sizeof (double) * N);
  
  // Generate initial conditions
  for (int i = 0; i < N; i++) {
    x[i] = (double)i * dx;
    u[i] = sin(x[i]);
  }

  // Allocate device memory
  cudaMalloc((void **)&u_dev, sizeof (double) * N);
  cudaMalloc((void **)&uu_dev, sizeof (double) * N);
  
  // Copy array to device memory
  if (error = cudaMemcpy(u_dev, u, sizeof (double) * N, cudaMemcpyHostToDevice))
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  cudaDeviceSynchronize();
  ///////////////////////////////////////////////////////////////////////////////////

  cufftDoubleComplex *uhat, *uuhat;

  //int fourier_mem_size = sizeof(cufftDoubleComplex)* (N/2 + 1);
  cudaMalloc((void**) &uhat, sizeof(cufftDoubleComplex) * (N/2 + 1) );
  cudaMalloc((void**) &uuhat, sizeof(cufftDoubleComplex) * (N/2 + 1) );
  
  // CUFFT plans
  cufftHandle plan_forward, plan_backward;
  cufftPlan1d(&plan_forward, N, CUFFT_D2Z, 1);
  cufftPlan1d(&plan_backward, N, CUFFT_Z2D, 1);

  // Get initial uhat 
  cufftExecD2Z(plan_forward, u_dev, uhat);
  // Set N/2th mode to zero
  kernel_setmodezero<<<1, 1>>>(uhat, N/2);
  cudaDeviceSynchronize();

  gettimeofday (&tdr0, NULL);

  // Main time stepping
  for (int i = 0; i < NADVANCE; i++) {

    // Fourier to real space
    cufftExecZ2D(plan_backward, uhat, u_dev);
    cudaDeviceSynchronize();

    // Rescale 
    kernel_rescale_u<<<nBlocks, blockSize>>>(u_dev, N);
    cudaDeviceSynchronize();

    // Calculate nonlinear product in real space
    kernel_calc_uu<<<nBlocks, blockSize>>>(u_dev, uu_dev);
    cudaDeviceSynchronize();

    // Nonlinear product in Fourier space
    cufftExecD2Z(plan_forward, uu_dev, uuhat);
    cudaDeviceSynchronize();
    kernel_setmodezero<<<1, 1>>>(uuhat, N/2);
    cudaDeviceSynchronize();
    
    // New uhat
    kernel_burgers<<<nBlocks/2, blockSize>>>(uhat, uuhat, dt);
    cudaDeviceSynchronize();
    kernel_setmodezero<<<1, 1>>>(uhat, N/2);
    cudaDeviceSynchronize();
  }
  
  // Final result
  cufftExecZ2D(plan_backward, uhat, u_dev);
  cudaDeviceSynchronize();
  
  /* retrieve results from device (synchronous) */
  if (error = cudaMemcpy(u, u_dev, sizeof (double) * N, cudaMemcpyDeviceToHost))
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  cudaDeviceSynchronize();
  
  // Rescale
  for (int i = 0; i < N; i++) {
    u[i] = u[i] / (double)N;
  }

  gettimeofday (&tdr1, NULL);

  // Timing information
  timeval_subtract (&restime, &tdr1, &tdr0);
  printf ("gpu time: %es\n", restime);

  // Show L2 norm
  for (int i = 0; i < N; i++) {
    norm += fabs(u[i] * u[i]);
  }
  norm = sqrt(norm);
  printf ("norm: %e\n", norm);

  // Print results to file
  if (out_file == NULL) {
    printf("Error opening file\n");
  }
  for (int i = 0; i < N; i++) {
    fprintf(out_file, "%.15f, %.15f\n", x[i], u[i]);
  }
  
  free(x);
  cudaFree(u);
  cudaFree(u_dev);
  cudaFree(uhat);
  cudaFree(uuhat);
  cufftDestroy(plan_forward);
  cufftDestroy(plan_backward);
  
  return 0;
}





