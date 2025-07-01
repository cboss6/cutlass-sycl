#include <cstdio>
#include <cuda_runtime.h>

__global__ void printThreadInfo() {
    printf(
        "GridDim=(%d,%d,%d)  "
        "BlockDim=(%d,%d,%d)  "
        "BlockIdx=(%d,%d,%d)  "
        "ThreadIdx=(%d,%d,%d)\n",
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        blockIdx.x, blockIdx.y, blockIdx.z,
        threadIdx.x, threadIdx.y, threadIdx.z
    );
}

int main() {
    // 1) 一维 grid + 一维 block，使用默认流， no shared memory
    //    Dg = dim3(10)       ⇒ gridDim.x = 10
    //    Db = dim3(5)        ⇒ blockDim.x = 5
    //    Ns = 0, S = 0
    printThreadInfo<<< dim3(10), dim3(5) >>>();
    cudaDeviceSynchronize();

    // 2) 二维 grid + 二维 block，分配 1 KB 动态共享内存，默认流
    //    Dg = dim3(4, 3)     ⇒ gridDim = {4,3,1}
    //    Db = dim3(8, 4)     ⇒ blockDim = {8,4,1}
    //    Ns = 1024 bytes, S = 0
    printThreadInfo<<< dim3(4, 3), dim3(8, 4), 1024 >>>();
    cudaDeviceSynchronize();

    // 3) 三维 grid + 三维 block，分配 2 KB 动态共享内存，使用自定义 stream
    //    Dg = dim3(2, 2, 2)
    //    Db = dim3(3, 3, 2)
    //    Ns = 2048 bytes, S = customStream
    cudaStream_t customStream;
    cudaStreamCreate(&customStream);

    printThreadInfo<<< dim3(2, 2, 2), dim3(3, 3, 2), 2048, customStream >>>();
    cudaStreamSynchronize(customStream);
    cudaStreamDestroy(customStream);

    return 0;
}
