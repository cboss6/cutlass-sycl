#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <random>

#define DEBUG
#ifdef DEBUG
#define DBG(format, ...) printf(format, ##__VA_ARGS__)
#else
#define DBG(format, ...)
#endif

template <typename T>
void generate_random_data(T* data, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dist(1, 100);
    for (int i = 0; i < size; ++i) {
        // data[i] = dist(gen);
        data[i] = i + 1;
    }
}

void accuracy_check(int* output, int* output_ref, int m, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (output[i * m + j] != output_ref[i * m + j]) {
                printf("Error at position (%d, %d): %d != %d\n", i, j, output[i * m + j], output_ref[i * m + j]);
                return;
            }
        }
    }
    printf("Accuracy check passed\n");
}

void transpose_cpu_ref(int *input, int m, int n, int *output) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            output[j * m + i] = input[i * n + j];
        }
    }
}
__global__ void transposeOptimized(int *input, int *output, int m, int n) {
    // 全局输入坐标
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 只处理合法范围内的线程
    if (row < m && col < n) {
        // 读全局到共享内存
        __shared__ int sdata[32][33];
        sdata[threadIdx.y][threadIdx.x] = input[row * n + col];
        __syncthreads();

        // 写回：交换行列
        int out_row = col;  // 原来的列
        int out_col = row;  // 原来的行
        // output[out_row * m + out_col] = sdata[threadIdx.x][threadIdx.y];
        output[out_row * m + out_col] = sdata[threadIdx.y][threadIdx.x];
    }
}


// __global__ void transposeOptimized(int *input, int *output, int m, int n) {
//     int colID_input = threadIdx.x + blockDim.x * blockIdx.x;
//     int rowID_input = threadIdx.y + blockDim.y * blockIdx.y;
    
//     __shared__ int sdata[32][33];
    
//     if (rowID_input < m && colID_input < n) {
//         int index_input = colID_input + rowID_input * n;
//         sdata[threadIdx.y][threadIdx.x] = input[index_input];
        
//         __syncthreads();
        
//         int dst_col = threadIdx.x + blockIdx.y * blockDim.y;
//         int dst_row = threadIdx.y + blockIdx.x * blockDim.x;       
//         output[dst_col + dst_row * m] = sdata[threadIdx.x][threadIdx.y];

//         DBG("blockDim: (%d, %d); blockIdx: (%d, %d); threadIdx: (%d, %d); inputGlobalIdx: (%d, %d); \
//              outputGlobalIdx: (%d, %d)\n", blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, colID_input, rowID_input, out_row, out_col);
//     }
// }

void printMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            DBG("%d ", matrix[i * cols + j]);
        }
        DBG("\n");
    }
    DBG("\n");
}

int main() {
    // 定義矩陣大小
    const int M = 4;  // 行數
    const int N = 3;  // 列數
    
    // 分配主機內存
    int *h_input = (int *)malloc(M * N * sizeof(int));
    int *h_output = (int *)malloc(M * N * sizeof(int));
    int *h_output_ref = (int *)malloc(M * N * sizeof(int));
    
    generate_random_data(h_input, M * N);
    DBG("Input Matrix (%dx%d):\n", M, N);
    printMatrix(h_input, M, N);
    transpose_cpu_ref(h_input, M, N, h_output_ref);
    DBG("Output Matrix (%dx%d):\n", N, M);
    printMatrix(h_output_ref, N, M);

    // 分配設備內存
    int *d_input, *d_output;
    cudaMalloc(&d_input, M * N * sizeof(int));
    cudaMalloc(&d_output, M * N * sizeof(int));
    
    // 將數據從主機複製到設備
    cudaMemcpy(d_input, h_input, M * N * sizeof(int), cudaMemcpyHostToDevice);
    
    // 設置線程塊和網格大小
    dim3 blockDim(16, 8);  // 每個線程塊4x4個線程
    DBG("blockDim: %d, %d\n", blockDim.x, blockDim.y);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);
    
    // 調用內核
    transposeOptimized<<<gridDim, blockDim>>>(d_input, d_output, M, N);
    
    // 將結果從設備複製回主機
    cudaMemcpy(h_output, d_output, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 打印輸入矩陣
    DBG("Input Matrix (%dx%d):\n", M, N);
    printMatrix(h_input, M, N);
    
    // 打印輸出矩陣（轉置後的結果）
    DBG("Output Matrix (%dx%d):\n", N, M);
    printMatrix(h_output, N, M);

    accuracy_check(h_output, h_output_ref, M, N);
    
    // 釋放內存
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}