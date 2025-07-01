// transpose_tiled_8192x4096.cu

#include <cstdio>
#include <cuda_runtime.h>

#define DEBUG
#ifdef DEBUG
#define DBG(format, ...) printf(format, ##__VA_ARGS__)
#else
#define DBG(format, ...)
#endif


#define TILE_DIM   32
#define BLOCK_ROWS 8

void printMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            DBG("%d ", matrix[i * cols + j]);
        }
        DBG("\n");
    }
    DBG("\n");
}

// GPU tiled transpose, no bank‐conflicts
__global__ void transposeOptimized(
    const int* __restrict__  input,
          int* __restrict__  output,
    int                     m,   // #rows of input
    int                     n)   // #cols of input
{
    __shared__ int tile[TILE_DIM][TILE_DIM + 1];
    //                        ↑ pad to TILE_DIM+1 to avoid bank conflicts

    // global coords in the input matrix
    int x_in = blockIdx.x * TILE_DIM + threadIdx.x;  // col index
    int y_in = blockIdx.y * TILE_DIM + threadIdx.y;  // row index

    // load one element into shared memory
    if (x_in < n && y_in < m) {
        tile[threadIdx.y][threadIdx.x] = input[y_in * n + x_in];
    }
    __syncthreads();

    // coordinates in the transposed (output) matrix:
    //   row_out = input column
    //   col_out = input row
    int row_out = blockIdx.x * TILE_DIM + threadIdx.y; // from x-block
    int col_out = blockIdx.y * TILE_DIM + threadIdx.x; // from y-block

    if (row_out < n && col_out < m) {
        // **note the order: row_out * m + col_out**
        output[row_out * m + col_out]
          = tile[threadIdx.x][threadIdx.y];
    }
    // blockDim: (32, 32); blockIdx: (0, 0); threadIdx: (0, 1); inputGlobalIdx: (0, 1) = tile: (0, 1) = 32; outputGlobalIdx: (1, 0) = tile: (1, 0) = 1
    // blockDim: (32, 32); blockIdx: (0, 0); threadIdx: (1, 1); inputGlobalIdx: (1, 1) = tile: (1, 1) = 33; outputGlobalIdx: (1, 1) = tile: (1, 1) = 33
    // blockDim: (32, 32); blockIdx: (0, 0); threadIdx: (2, 1); inputGlobalIdx: (2, 1) = tile: (2, 1) = 34; outputGlobalIdx: (1, 2) = tile: (1, 2) = 65
    // blockDim: (32, 32); blockIdx: (0, 0); threadIdx: (3, 1); inputGlobalIdx: (3, 1) = tile: (3, 1) = 35; outputGlobalIdx: (1, 3) = tile: (1, 3) = 97
    // blockDim: (32, 32); blockIdx: (0, 0); threadIdx: (4, 1); inputGlobalIdx: (4, 1) = tile: (4, 1) = 36; outputGlobalIdx: (1, 4) = tile: (1, 4) = 129
    // blockDim: (32, 32); blockIdx: (0, 0); threadIdx: (5, 1); inputGlobalIdx: (5, 1) = tile: (5, 1) = 37; outputGlobalIdx: (1, 5) = tile: (1, 5) = 161
    // blockDim: (32, 32); blockIdx: (0, 0); threadIdx: (6, 1); inputGlobalIdx: (6, 1) = tile: (6, 1) = 38; outputGlobalIdx: (1, 6) = tile: (1, 6) = 193
    // blockDim: (32, 32); blockIdx: (0, 0); threadIdx: (7, 1); inputGlobalIdx: (7, 1) = tile: (7, 1) = 39; outputGlobalIdx: (1, 7) = tile: (1, 7) = 225
    // blockDim: (32, 32); blockIdx: (0, 0); threadIdx: (8, 1); inputGlobalIdx: (8, 1) = tile: (8, 1) = 40; outputGlobalIdx: (1, 8) = tile: (1, 8) = 257
    // blockDim: (32, 32); blockIdx: (0, 0); threadIdx: (9, 1); inputGlobalIdx: (9, 1) = tile: (9, 1) = 41; outputGlobalIdx: (1, 9) = tile: (1, 9) = 289
    DBG("blockDim: (%d, %d); blockIdx: (%d, %d); threadIdx: (%d, %d); tile: (%d, %d) = inputGlobalIdx: (%d, %d) = %d; \
        outputGlobalIdx: (%d, %d) = tile: (%d, %d) = %d\n", blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, 
        threadIdx.x, threadIdx.y, threadIdx.x, threadIdx.y, x_in, y_in,  input[y_in * n + x_in], col_out, row_out, threadIdx.y, threadIdx.x, output[row_out * m + col_out]);
}

// simple CPU reference
void cpuTranspose(const int *in, int *out, int m, int n) {
    // produces out[c * m + r] = in[r * n + c]
    for (int r = 0; r < m; ++r)
        for (int c = 0; c < n; ++c)
            out[c * m + r] = in[r * n + c];
}

int main() {
    // const int M = 8192;
    // const int N = 4096;
    const int M = 64;
    const int N = 32;
    size_t    SIZE = size_t(M) * size_t(N) * sizeof(int);

    // host allocations
    int *h_in  = (int*)malloc(SIZE);
    int *h_out = (int*)malloc(SIZE);
    int *h_ref = (int*)malloc(SIZE);

    // initialize input
    for (size_t i = 0; i < size_t(M)*N; ++i) {
        h_in[i] = int(i);
    }

    printMatrix(h_in, M, N);

    // compute reference on CPU
    cpuTranspose(h_in, h_ref, M, N);

    // device allocations
    int *d_in, *d_out;
    cudaMalloc(&d_in,  SIZE);
    cudaMalloc(&d_out, SIZE);
    cudaMemcpy(d_in, h_in, SIZE, cudaMemcpyHostToDevice);

    // launch
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM,
              (M + TILE_DIM - 1) / TILE_DIM);

    transposeOptimized<<<grid, block>>>(d_in, d_out, M, N);
    cudaDeviceSynchronize();

    // copy back
    cudaMemcpy(h_out, d_out, SIZE, cudaMemcpyDeviceToHost);

    // verify
    bool pass = true;
    for (size_t i = 0; i < size_t(M)*N; ++i) {
        if (h_out[i] != h_ref[i]) {
            printf("Mismatch at idx %zu: gpu=%d  cpu=%d\n",
                   i, h_out[i], h_ref[i]);
            pass = false;
            break;
        }
    }
    printf(pass ? "PASS\n" : "FAIL\n");

    printMatrix(h_out, N, M);

    // cleanup
    free(h_in); free(h_out); free(h_ref);
    cudaFree(d_in); cudaFree(d_out);
    return pass ? 0 : 1;
}
