#include <stdio.h>
#include <cuda_runtime.h>

void printDeviceInfo(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("\nDevice %d: \"%s\"\n", device, prop.name);
    printf("  Total global memory:                 %zu bytes\n", prop.totalGlobalMem);
    printf("  Shared memory per block:             %zu bytes\n", prop.sharedMemPerBlock);
    printf("  Registers per block:                 %d\n", prop.regsPerBlock);
    printf("  Warp size:                           %d\n", prop.warpSize);
    printf("  Maximum threads per block:           %d\n", prop.maxThreadsPerBlock);
    printf("  Maximum block dimensions:            %d x %d x %d\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Maximum grid dimensions:             %d x %d x %d\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Clock rate:                          %.2f GHz\n", prop.clockRate * 1e-6f);
    printf("  Total constant memory:               %zu bytes\n", prop.totalConstMem);
    printf("  Compute capability:                  %d.%d\n", prop.major, prop.minor);
    printf("  Number of multiprocessors:           %d\n", prop.multiProcessorCount);
    printf("  Concurrent kernels:                  %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("  Device overlap:                      %s\n", prop.deviceOverlap ? "Yes" : "No");
    printf("  Memory clock rate:                   %.2f GHz\n", prop.memoryClockRate * 1e-6f);
    printf("  Memory bus width:                    %d bits\n", prop.memoryBusWidth);
    printf("  L2 cache size:                       %d bytes\n", prop.l2CacheSize);
    printf("  Max threads per multiprocessor:      %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Unified addressing:                  %s\n", prop.unifiedAddressing ? "Yes" : "No");
    printf("  ECC enabled:                         %s\n", prop.ECCEnabled ? "Yes" : "No");
    printf("  PCI bus ID:                          %d\n", prop.pciBusID);
    printf("  PCI device ID:                       %d\n", prop.pciDeviceID);
    printf("  PCI domain ID:                       %d\n", prop.pciDomainID);
    printf("  TCC driver:                          %s\n", prop.tccDriver ? "Yes" : "No");
    printf("  Async engine count:                  %d\n", prop.asyncEngineCount);
    printf("  Kernel execution timeout:            %s\n", prop.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf("  Integrated GPU sharing host memory:  %s\n", prop.integrated ? "Yes" : "No");
    printf("  Can map host memory:                 %s\n", prop.canMapHostMemory ? "Yes" : "No");
    printf("  Compute mode:                        %d\n", prop.computeMode);
    printf("  Maximum texture 1D size:             %d\n", prop.maxTexture1D);
    printf("  Maximum texture 2D size:             %d x %d\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
    printf("  Maximum texture 3D size:             %d x %d x %d\n", 
           prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
    printf("  Maximum texture 1D layered size:     %d x %d\n", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]);
    printf("  Maximum texture 2D layered size:     %d x %d x %d\n", 
           prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);
    printf("  Surface alignment:                   %zu\n", prop.surfaceAlignment);
    printf("  Concurrent copy and kernel execution:%s\n", prop.deviceOverlap ? "Yes" : "No");
    printf("  Number of async engines:             %d\n", prop.asyncEngineCount);
    printf("  Device supports host page-locked memory mapping: %s\n", prop.canMapHostMemory ? "Yes" : "No");
    printf("  Device supports unified addressing:  %s\n", prop.unifiedAddressing ? "Yes" : "No");
    printf("  Device PCI Domain ID / Bus ID / location ID: %d / %d / %d\n", 
           prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("Found %d CUDA devices\n", deviceCount);
    
    for (int device = 0; device < deviceCount; device++) {
       int max_regs_per_block = 0;
       int max_shmem_per_block = 0;
       cudaDeviceGetAttribute(&max_regs_per_block, cudaDevAttrMaxRegistersPerBlock, device);
       printf("max_regs_per_block: %d\n", max_regs_per_block);
       cudaDeviceGetAttribute(&max_shmem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device);
       printf("max_shmem_per_block: %d\n", max_shmem_per_block);
        printDeviceInfo(device);
    }
    
    return 0;
} 