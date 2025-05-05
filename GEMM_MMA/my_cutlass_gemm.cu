// file: managed_demo.cu
#include <cuda_runtime.h>
#include <iostream>

// 在全局区声明一个 managed 数组
__managed__ int data[10];

// 一个简单的 kernel：每个线程把 data[i] 加上 i
__global__ void add_index() {
    int i = threadIdx.x;
    if (i < 10) {
        data[i] += i;
    }
}

int main() {
    // 1. 在 host 上初始化
    for (int i = 0; i < 10; i++) {
        data[i] = i * 100;
    }

    // 2. 启动 kernel，用 1 个 block，10 个 thread
    add_index<<<1, 10>>>();
    cudaDeviceSynchronize();  // 等待 GPU 完成

    // 3. 直接在 host 上读 managed 内存并打印
    std::cout << "Result after kernel:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "data[" << i << "] = " << data[i] << std::endl;
    }
    return 0;
}
