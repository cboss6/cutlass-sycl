#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <float.h>

// --------------------------------------------------------
// Step 1: Initialize padded arrays (values + indices)
// --------------------------------------------------------
__global__ void init_arrays(const float* d_input, int N,
                            float* vals, int* idxs, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        if (i < N) {
            vals[i] = d_input[i];
            idxs[i] = i;
        } else {
            vals[i] = -FLT_MAX;  // pad small value
            idxs[i] = -1;
        }
    }
}

// --------------------------------------------------------
// Step 2: Bitonic sort step for descending order
// --------------------------------------------------------
__global__ void bitonic_step(float* vals, int* idxs, int M, int j, int k) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    unsigned int ixj = i ^ j;
    if (ixj > i) {
        // decide order based on bit k
        if ((i & k) == 0) {
            // ascending stage -> we want larger first => swap if vals[i] < vals[ixj]
            if (vals[i] < vals[ixj]) {
                float tmp = vals[i]; vals[i] = vals[ixj]; vals[ixj] = tmp;
                int it  = idxs[i]; idxs[i] = idxs[ixj]; idxs[ixj] = it;
            }
        } else {
            // descending stage -> swap if vals[i] > vals[ixj]
            if (vals[i] > vals[ixj]) {
                float tmp = vals[i]; vals[i] = vals[ixj]; vals[ixj] = tmp;
                int it  = idxs[i]; idxs[i] = idxs[ixj]; idxs[ixj] = it;
            }
        }
    }
}

// --------------------------------------------------------
// Host function: Perform global bitonic sort and extract Top-K
// --------------------------------------------------------
void topk_bitonic(const float* d_input, int N, int K,
                  float* d_out_vals, int* d_out_idxs) {
    // Round up to next power-of-two
    int M = 1;
    while (M < N) M <<= 1;

    // Allocate padded arrays
    float* d_vals;
    int*   d_idxs;
    cudaMalloc(&d_vals, M * sizeof(float));
    cudaMalloc(&d_idxs, M * sizeof(int));

    // Init values + indices
    int threads = 256;
    int blocks  = (M + threads - 1) / threads;
    init_arrays<<<blocks, threads>>>(d_input, N, d_vals, d_idxs, M);
    cudaDeviceSynchronize();

    // Bitonic sort: k = 2,4,8,...,M; j = k/2,k/4,...,1
    for (int k = 2; k <= M; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_step<<<blocks, threads>>>(d_vals, d_idxs, M, j, k);
            cudaDeviceSynchronize();
        }
    }

    // Copy Top-K to output
    cudaMemcpy(d_out_vals, d_vals, K * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_out_idxs, d_idxs, K * sizeof(int),   cudaMemcpyDeviceToDevice);

    cudaFree(d_vals);
    cudaFree(d_idxs);
}

// --------------------------------------------------------
// Example usage
// --------------------------------------------------------
int main() {
    const int N = 16;
    const int K = 5;
    float h_in[N] = { 1.2f, 9.5f, 0.1f, 3.3f, 7.7f, 2.2f, 8.8f, 5.5f,
                      4.4f, 6.6f, 1.1f, 9.9f, 0.2f, 3.7f, 7.1f, 2.8f };

    // Device buffers
    float *d_in, *d_out_vals;
    int   *d_out_idxs;
    cudaMalloc(&d_in,        N * sizeof(float));
    cudaMalloc(&d_out_vals,  K * sizeof(float));
    cudaMalloc(&d_out_idxs,  K * sizeof(int));

    // Copy input
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // Compute Top-K
    topk_bitonic(d_in, N, K, d_out_vals, d_out_idxs);

    // Copy and print results
    std::vector<float> h_vals(K);
    std::vector<int>   h_idxs(K);
    cudaMemcpy(h_vals.data(), d_out_vals, K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idxs.data(), d_out_idxs, K * sizeof(int),   cudaMemcpyDeviceToHost);

    std::cout << "Top-" << K << " values and indices (descending):\n";
    for (int i = 0; i < K; i++) {
        std::cout << "  " << h_vals[i]
                  << " (idx=" << h_idxs[i] << ")\n";
    }

    cudaFree(d_in);
    cudaFree(d_out_vals);
    cudaFree(d_out_idxs);
    return 0;
}
