# PCA-EXP-6-MATRIX-TRANSPOSITION-USING-SHARED-MEMORY-AY-23-24
<h3>AIM:To perform Matrix Multiplication using Transposition using shared memory.</h3>
<h3>RIHAB ZAKKAIR HUSSAIN</h3>
<h3>212225230226</h3>
<h3>EX. NO :- 6</h3>
<h3>31-8-2026</h3>
<h1> <align=center> MATRIX TRANSPOSITION USING SHARED MEMORY </h3>
  Implement Matrix transposition using GPU Shared memory.</h3>

## AIM:
To perform Matrix Multiplication using Transposition using shared memory.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler

## PROCEDURE:
 CUDA_SharedMemory_AccessPatterns:

1. Begin Device Setup
    1.1 Select the device to be used for computation
    1.2 Retrieve the properties of the selected device
2. End Device Setup

3. Begin Array Size Setup
    3.1 Set the size of the array to be used in the computation
    3.2 The array size is determined by the block dimensions (BDIMX and BDIMY)
4. End Array Size Setup

5. Begin Execution Configuration
    5.1 Set up the execution configuration with a grid and block dimensions
    5.2 In this case, a single block grid is used
6. End Execution Configuration

7. Begin Memory Allocation
    7.1 Allocate device memory for the output array d_C
    7.2 Allocate a corresponding array gpuRef in the host memory
8. End Memory Allocation

9. Begin Kernel Execution
    9.1 Launch several kernel functions with different shared memory access patterns (Use any two patterns)
        9.1.1 setRowReadRow: Each thread writes to and reads from its row in shared memory
        9.1.2 setColReadCol: Each thread writes to and reads from its column in shared memory
        9.1.3 setColReadCol2: Similar to setColReadCol, but with transposed coordinates
        9.1.4 setRowReadCol: Each thread writes to its row and reads from its column in shared memory
        9.1.5 setRowReadColDyn: Similar to setRowReadCol, but with dynamic shared memory allocation
        9.1.6 setRowReadColPad: Similar to setRowReadCol, but with padding to avoid bank conflicts
        9.1.7 setRowReadColDynPad: Similar to setRowReadColPad, but with dynamic shared memory allocation
10. End Kernel Execution

11. Begin Memory Copy
    11.1 After each kernel execution, copy the output array from device memory to host memory
12. End Memory Copy

13. Begin Memory Free
    13.1 Free the device memory and host memory
14. End Memory Free

15. Reset the device

16. End of Algorithm

## PROGRAM:
```
%%writefile shared_memory_transpose_clean.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#pragma diag_suppress 177
#pragma diag_suppress 20012

#define CHECK(call)                                                     \
{                                                                       \
    const cudaError_t error = call;                                     \
    if (error != cudaSuccess)                                           \
    {                                                                   \
        printf("Error: %s:%d, code:%d, reason:%s\n",                    \
            __FILE__, __LINE__, error, cudaGetErrorString(error));      \
        exit(1);                                                        \
    }                                                                   \
}

#define BDIMX 16
#define BDIMY 16
#define IPAD  2

void printData(const char *msg, int *in, int size)
{
    printf("%s: ", msg);
    for (int i = 0; i < size; i++)
        printf("%4d", in[i]);
    printf("\n\n");
}

// ----------------------------------------------------------
//  KERNELS
// ----------------------------------------------------------

__global__ void setRowReadRow(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int *out)
{
    __shared__ int tile[BDIMX][BDIMY];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadCol2(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    tile[icol][irow] = idx;
    __syncthreads();
    out[idx] = tile[icol][irow];
}

__global__ void setRowReadCol(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[icol][irow];
}

__global__ void setRowReadColPad(int *out)
{
    __shared__ int tile[BDIMY][BDIMX + IPAD];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[icol][irow];
}

__global__ void setRowReadColDyn(int *out)
{
    extern __shared__ int tile[];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    unsigned int col_idx = icol * blockDim.x + irow;

    tile[idx] = idx;
    __syncthreads();
    out[idx] = tile[col_idx];
}

__global__ void setRowReadColDynPad(int *out)
{
    extern __shared__ int tile[];
    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int irow = g_idx / blockDim.y;
    unsigned int icol = g_idx % blockDim.y;

    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int col_idx = icol * (blockDim.x + IPAD) + irow;

    tile[row_idx] = g_idx;
    __syncthreads();
    out[g_idx] = tile[col_idx];
}

// ----------------------------------------------------------
// MAIN
// ----------------------------------------------------------

int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    CHECK(cudaSetDevice(dev));

    printf("Running on device: %s\n", prop.name);

    int nx = BDIMX, ny = BDIMY;
    int size = nx * ny;
    size_t nBytes = size * sizeof(int);

    bool printFlag = true;
    if (argc > 1) printFlag = atoi(argv[1]);

    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);

    printf("<<< grid (%d,%d), block (%d,%d) >>>\n\n",
           grid.x, grid.y, block.x, block.y);

    int *d_C, *gpuRef;
    CHECK(cudaMalloc((void**)&d_C, nBytes));
    gpuRef = (int*)malloc(nBytes);

    // Run each kernel
    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadRow<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (printFlag) printData("setRowReadRow", gpuRef, size);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setColReadCol<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (printFlag) printData("setColReadCol", gpuRef, size);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setColReadCol2<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (printFlag) printData("setColReadCol2", gpuRef, size);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadCol<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (printFlag) printData("setRowReadCol", gpuRef, size);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColDyn<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (printFlag) printData("setRowReadColDyn", gpuRef, size);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColPad<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (printFlag) printData("setRowReadColPad", gpuRef, size);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColDynPad<<<grid, block,
                         (BDIMX + IPAD) * BDIMY * sizeof(int)>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (printFlag) printData("setRowReadColDynPad", gpuRef, size);

    free(gpuRef);
    CHECK(cudaFree(d_C));
    CHECK(cudaDeviceReset());

    return 0;
}


!nvcc -arch=sm_75 shared_memory_transpose_clean.cu -o shared_mem
!./shared_mem
```
## OUTPUT:
<img width="1465" height="351" alt="image" src="https://github.com/user-attachments/assets/57d65a11-5c98-41b3-8cba-32483068e8bb" />


## RESULT:
Thus the program has been executed by using CUDA to transpose a matrix. It is observed that there are variations shared memory and global memory implementation. The elapsed times are recorded as _______________.
