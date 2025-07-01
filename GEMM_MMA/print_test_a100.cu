#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include "cutlass/util/print_error.hpp"

template <class TensorS, class TensorD, class ThreadLayout, class VecLayout>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, ThreadLayout, VecLayout)
{
  using namespace cute;
  using Element = typename TensorS::value_type;

  Tensor tile_S = S(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)

  using AccessType = cutlass::AlignedArray<Element, size(VecLayout{})>;
  using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;

  auto tiled_copy =
    make_tiled_copy(
      Atom{},                       // access size
      ThreadLayout{},               // thread layout
      VecLayout{});                 // vector layout (e.g. 4x1)
  
  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

  Tensor thr_tile_S = thr_copy.partition_S(tile_S);             // (CopyOp, CopyM, CopyN)
  if (thread0()) {
    printf("thr_tile_S:\n");
    cute::print_tensor(thr_tile_S);
  }
  Tensor thr_tile_D = thr_copy.partition_D(tile_D);             // (CopyOp, CopyM, CopyN)
  Tensor fragment = make_fragment_like(thr_tile_D);             // (CopyOp, CopyM, CopyN)

  copy(tiled_copy, thr_tile_S, fragment);
  copy(tiled_copy, fragment, thr_tile_D);
}

/// Main function
int main(int argc, char** argv)
{
  using namespace cute;
  using Element = float;

  auto tensor_shape = make_shape(256, 128);

  std::vector<Element> h_S(size(tensor_shape));
  std::vector<Element> h_D(size(tensor_shape));

  Element *d_S, *d_D;
  cudaMalloc(&d_S, size(tensor_shape) * sizeof(Element));
  cudaMalloc(&d_D, size(tensor_shape) * sizeof(Element));

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
  }

  cudaMemcpy(d_S, h_S.data(), size(tensor_shape) * sizeof(Element), cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, h_D.data(), size(tensor_shape) * sizeof(Element), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  Tensor tensor_S = make_tensor(d_S, make_layout(tensor_shape));
  Tensor tensor_D = make_tensor(d_D, make_layout(tensor_shape));

  auto block_shape = make_shape(Int<128>{}, Int<64>{});

  // check if the tensor shape is divisible by the block shape
  if ((size<0>(tensor_shape) % size<0>(block_shape)) || (size<1>(tensor_shape) % size<1>(block_shape))) {
    std::cerr << "The tensor shape must be divisible by the block shape." << std::endl;
    return -1;
  }

  // Equivalent check to the above
  if (not evenly_divides(tensor_shape, block_shape)) {
    std::cerr << "Expected the block_shape to evenly divide the tensor shape." << std::endl;
    return -1;
  }

  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);      // ((M, N), m', n')
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);      // ((M, N), m', n')

  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));
  Layout vec_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));

  dim3 gridDim(size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));  // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(thr_layout));

  copy_kernel_vectorized<decltype(tiled_tensor_S), decltype(tiled_tensor_D),
                        decltype(thr_layout), decltype(vec_layout)><<<gridDim, blockDim>>>(
      tiled_tensor_S, tiled_tensor_D, thr_layout, vec_layout);
  cudaDeviceSynchronize();
  return 0;
}
