/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"

#define THREAD_COND (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)

template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout)
{
  using namespace cute;

  // Slice the tiled tensors
  Tensor tile_S = S(make_coord(_,_), blockIdx.x,
                    blockIdx.y);            // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_,_), blockIdx.x,
                    blockIdx.y);            // (BlockShape_M, BlockShape_N)

  // Construct a partitioning of the tile among threads with the given thread arrangement.

  // Concept:                         Tensor  ThrLayout       ThrIndex
  Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)
  Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)

  // Construct a register-backed Tensor with the same shape as each thread's partition
  // Use make_tensor to try to match the layout of thr_tile_S
  Tensor fragment = make_tensor_like(thr_tile_S);               // (ThrValM, ThrValN)

  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy(thr_tile_S, fragment);
  copy(fragment, thr_tile_D);
}

/// Vectorized copy kernel.
///
/// Uses `make_tiled_copy()` to perform a copy using vector instructions. This operation
/// has the precondition that pointers are aligned to the vector size.
///
template <class TensorS, class TensorD, class ThreadLayout, class VecLayout>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, ThreadLayout, VecLayout)
{
  using namespace cute;
  using Element = typename TensorS::value_type;

  DBG_IF(THREAD_COND, "blockIdx.x = %d, blockIdx.y = %d, threadIdx.x = %d\n", blockIdx.x, blockIdx.y, threadIdx.x);
  // Slice the tensors to obtain a view into each tile.
  DBG_IF(THREAD_COND, "S.layout():\n");
  if(THREAD_COND) DBG_(S.layout()); DBG_IF(THREAD_COND, "\n");
  Tensor tile_S = S(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)
  if (THREAD_COND) {
    DBG("tile_S.layout():\n");
    DBG_(tile_S.layout()); DBG("\n");
    DBG_TENSOR(tile_S); DBG("\n");
  }

  // Define `AccessType` which controls the size of the actual memory access.
  using AccessType = cutlass::AlignedArray<Element, size(VecLayout{})>;

  // A copy atom corresponds to one hardware memory access.
  using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;

  // Construct tiled copy, a tiling of copy atoms.
  auto tiled_copy =
    make_tiled_copy(
      Atom{},                       // access size
      ThreadLayout{},               // thread layout
      VecLayout{});                 // vector layout (e.g. 4x1)
  if (THREAD_COND) {
    DBG("tiled_copy:\n");
    DBG_(tiled_copy); DBG("\n");
    // print_latex(tiled_copy);
  }
  
  // Construct a Tensor corresponding to each thread's slice.
  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
  if (THREAD_COND) {
    DBG("thr_copy:\n");
    DBG_(thr_copy); DBG("\n");
    // print_latex(thr_copy);
  }

  Tensor thr_tile_S = thr_copy.partition_S(tile_S);             // (CopyOp, CopyM, CopyN)
  if (THREAD_COND) {
    DBG("thr_tile_S:\n");
    DBG_(thr_tile_S.layout()); DBG("\n");
    DBG_TENSOR(thr_tile_S); DBG("\n");
  }
  Tensor thr_tile_D = thr_copy.partition_D(tile_D);             // (CopyOp, CopyM, CopyN)
  if (THREAD_COND) {
    DBG("thr_tile_D before:\n");
    DBG_(thr_tile_D.layout()); DBG("\n");
    DBG_TENSOR(thr_tile_D); DBG("\n");
  }

  // Construct a register-backed Tensor with the same shape as each thread's partition
  Tensor fragment = make_fragment_like(thr_tile_D);             // (CopyOp, CopyM, CopyN)
  if (THREAD_COND) {
    DBG("fragment before:\n");
    DBG_(fragment.layout()); DBG("\n");
    DBG_TENSOR(fragment); DBG("\n");
  }

  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy(tiled_copy, thr_tile_S, fragment);
  if (THREAD_COND) {
    DBG("fragment after:\n");
    DBG_(fragment.layout()); DBG("\n");
    DBG_TENSOR(fragment); DBG("\n");
  }
  copy(tiled_copy, fragment, thr_tile_D);
  if (THREAD_COND) {
    DBG("thr_tile_D after:\n");
    DBG_(thr_tile_D.layout()); DBG("\n");
    DBG_TENSOR(thr_tile_D); DBG("\n");
  }
}

/// Main function
int main(int argc, char** argv)
{
  //
  // Given a 2D shape, perform an efficient copy
  //
  std::cout << "CBOSS" << std::endl;
  using namespace cute;
  using Element = float;

  // Define a tensor shape with dynamic extents (m, n)
  auto tensor_shape = make_shape(256, 128);

  //
  // Allocate and initialize
  //
  std::vector<Element> h_S(size(tensor_shape));
  std::vector<Element> h_D(size(tensor_shape));

  Element *d_S, *d_D;
  cudaMalloc(&d_S, size(tensor_shape) * sizeof(Element));
  cudaMalloc(&d_D, size(tensor_shape) * sizeof(Element));

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
    h_D[i] = -static_cast<Element>(i);
  }

  cudaMemcpy(d_S, h_S.data(), size(tensor_shape) * sizeof(Element), cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, h_D.data(), size(tensor_shape) * sizeof(Element), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  //
  // Make tensors
  //
  Tensor tensor_S = make_tensor(d_S, make_layout(tensor_shape));
  Tensor tensor_D = make_tensor(d_D, make_layout(tensor_shape));

  //
  // Tile tensors
  //

  // Define a statically sized block (M, N).
  auto block_shape = make_shape(Int<128>{}, Int<64>{});

  if ((size<0>(tensor_shape) % size<0>(block_shape)) || (size<1>(tensor_shape) % size<1>(block_shape))) {
    std::cerr << "The tensor shape must be divisible by the block shape." << std::endl;
    return -1;
  }
  if (not evenly_divides(tensor_shape, block_shape)) {
    std::cerr << "Expected the block_shape to evenly divide the tensor shape." << std::endl;
    return -1;
  }

  // Tile the tensor (m, n) ==> ((M, N), m', n')
  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);      // ((M, N), m', n')
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);      // ((M, N), m', n')
  DBG("CBOSS size<0>(tiled_tensor_S)=%d, size<1>(tiled_tensor_S)=%d, size<2>(tiled_tensor_S)=%d\n", size<0>(tiled_tensor_S)(), size<1>(tiled_tensor_S), size<2>(tiled_tensor_S));

  // Thread arrangement
  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));
  // Vector dimensions
  Layout vec_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));

  //
  // Determine grid and block dimensions
  //
  dim3 gridDim(size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));  // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(thr_layout));
  DBG("gridDim.x=%d, gridDim.y=%d, blockDim.x=%d, blockDim.y=%d\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

  //
  // Launch the kernel
  //
  copy_kernel_vectorized<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_D, thr_layout, vec_layout);
  cudaDeviceSynchronize();

  //
  // Verify
  //
  cudaMemcpy(h_D.data(), d_D, size(tensor_shape) * sizeof(Element), cudaMemcpyDeviceToHost);

  int32_t errors = 0;
  int32_t const kErrorLimit = 10;

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_S[i] != h_D[i]) {
      std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;

      if (++errors >= kErrorLimit) {
        std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
        return -1;
      }
    }
  }

  std::cout << "Success." << std::endl;

  // Cleanup
  cudaFree(d_S);
  cudaFree(d_D);

  return 0;
}
 