// file: managed_demo.cu
#include <cuda_runtime.h>
#include <iostream>

#include <cutlass/util/host_tensor.h>
#include <cutlass/layout/matrix.h>

void test_column_major_tensor() {
    // 定义矩阵维度
    int const M = 3;  // 行数
    int const N = 4;  // 列数

    // 创建Column Major tensor
    cutlass::HostTensor<int, cutlass::layout::ColumnMajor> tensor(
        cutlass::MatrixCoord(M, N)
    );

    // 初始化数据
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            tensor.at(cutlass::MatrixCoord(m, n)) = m * N + n;
        }
    }

    // 验证数据
    // tensor.sync_host();
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            // 在Column Major中，元素在内存中的位置是 m + n * M
            // assert(tensor.at(cutlass::MatrixCoord(m, n)) == m * N + n);
            printf("tensor[%d, %d] = %d, ", m, n, tensor.at(cutlass::MatrixCoord(m, n)));
        }
        printf("\n");
    }
    for (int i = 0; i < M*N; i++) {
        printf("%d, ", tensor.host_data()[i]);
        if ((i + 1) % N == 0) printf("\n");
    }
    printf("\n");
}

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
    // add_index<<<1, 10>>>();
    // cudaDeviceSynchronize();  // 等待 GPU 完成

    // 3. 直接在 host 上读 managed 内存并打印
    // std::cout << "Result after kernel:" << std::endl;
    // for (int i = 0; i < 10; i++) {
    //     std::cout << "data[" << i << "] = " << data[i] << std::endl;
    // }

    test_column_major_tensor();
    return 0;
}
