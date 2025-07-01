#include <stdio.h>
#include <stdlib.h>

__global__ void transposeOptimized(float *input, float *output, int m, int n) {
    int colID_input = threadIdx.x + blockDim.x * blockIdx.x;
    int rowID_input = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (rowID_input < m && colID_input < n) {
        // 直接轉置：將行和列互換
        output[colID_input + rowID_input * m] = input[rowID_input + colID_input * n];
    }
}

void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.1f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // 定義矩陣大小
    const int M = 4;  // 行數
    const int N = 4;  // 列數
    
    // 分配主機內存
    float *h_input = (float *)malloc(M * N * sizeof(float));
    float *h_output = (float *)malloc(M * N * sizeof(float));
    
    // 初始化輸入矩陣
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_input[i * N + j] = i * N + j + 1;  // 填充1到16的數字
        }
    }
    
    // 分配設備內存
    float *d_input, *d_output;
    cudaMalloc(&d_input, M * N * sizeof(float));
    cudaMalloc(&d_output, M * N * sizeof(float));
    
    // 將數據從主機複製到設備
    cudaMemcpy(d_input, h_input, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 設置線程塊和網格大小
    dim3 blockDim(4, 4);  // 每個線程塊4x4個線程
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);
    
    // 調用內核
    transposeOptimized<<<gridDim, blockDim>>>(d_input, d_output, M, N);
    
    // 將結果從設備複製回主機
    cudaMemcpy(h_output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 打印輸入矩陣
    printf("Input Matrix (%dx%d):\n", M, N);
    printMatrix(h_input, M, N);
    
    // 打印輸出矩陣（轉置後的結果）
    printf("Output Matrix (%dx%d):\n", N, M);
    printMatrix(h_output, N, M);
    
    // 釋放內存
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
} 