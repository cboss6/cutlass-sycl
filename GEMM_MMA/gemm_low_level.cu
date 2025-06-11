#include <cuda_runtime.h>
#include <cuda.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/cutlass.h"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/GPU_Clock.hpp"
#include <cute/tensor.hpp>
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_elementwise.h"
#include "cutlass/util/tensor_view_io.h"
#include "cute/atom/copy_atom.hpp"
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_traits.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#include <functional>
#include <iostream>
#include <stdio.h>
#define ENABLE_CUTLASS 1
#define ENABLE_DEBUG 1

template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage
{
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta)
{
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles across the threads
  //

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

  //
  // PREFETCH
  //

  auto K_PIPE_MAX = size<3>(tAsA);

  // Total count of tiles
  int k_tile_count = size<3>(tAgA);
  // Current tile index in gmem to read from
  int k_tile_next = 0;

  // Start async loads for all pipes but the last
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) { ++k_tile_next; }
  }

  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));               // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V((  shape(tCrC) == take<0,3>(shape(tCgC))));     // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA)));              // MMA_M
  CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB)));              // MMA_N

  // Clear the accumulators
  clear(tCrC);

  //
  // Copy Atom retiling
  //

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K,PIPE)
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                         // (CPY,MMA_N,MMA_K)

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrA : "); print(tCrA); print("\n");
    print("tCrB : "); print(tCrB); print("\n");
    print("tCrC : "); print(tCrC); print("\n");

    print("tXsA : "); print(tXsA); print("\n");
    print("tXrA : "); print(tXrA); print("\n");
    print("tXsB : "); print(tXsB); print("\n");
    print("tXrB : "); print(tXrB); print("\n");
  }
#endif

#if 1

  // Current pipe index in smem to read from
  int smem_pipe_read  = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = K_PIPE_MAX-1;

  // Pipe slice
  Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
  Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

  // Size of the register pipeline
  auto K_BLOCK_MAX = size<2>(tCrA);

  // PREFETCH register pipeline
  if (K_BLOCK_MAX > 1) {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<K_PIPE_MAX-2>();
    __syncthreads();

    // Prefetch the first rmem from the first k-tile
    copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
  }

  //
  // PIPELINED MAIN LOOP
  // TUTORIAL: Example of a gemm loop that pipelines shared memory using SM80's cp.async instructions
  //           and explicit pipelines in shared memory.
  //   Data is read from global(k_tile_next) to shared(smem_pipe_write).
  //   Data is read from shared(smem_pipe_read) to registers(k_block_next).
  //   Data is computed on registers(b_block).
  //
  //   This allows all copies and compute to overlap:
  //     Copy from gmem->smem can overlap with copies from smem->rmem and compute on rmem.
  //     Copy from smem->rmem can overlap with compute on rmem.
  //

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX-1))
  {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        // Slice the smem_pipe_read smem
        tXsA_p = tXsA(_,_,_,smem_pipe_read);
        tXsB_p = tXsB(_,_,_,smem_pipe_read);

        // Commit the smem for smem_pipe_read
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static
      copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
      copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));
      // Copy gmem to smem before computing gemm on each k-pipe
      if (k_block == 0)
      {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        cp_async_fence();

        // Advance the gmem tile
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }

        // Advance the smem pipe
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }

  }

#endif

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);
}

template <class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        cute::half_t const* A, int ldA,
        cute::half_t const* B, int ldB,
        Beta beta,
        cute::half_t      * C, int ldC,
        cudaStream_t stream = 0)
{
  assert(false && "Not implemented");
}

// Setup params for a TN HGEMM
template <class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        cute::half_t const* A, int ldA,
        cute::half_t const* B, int ldB,
        Beta beta,
        cute::half_t      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)
  DBG("dA is: ");
  DBG_(dA); DBG("\n");
  DBG("dB is: ");
  DBG_(dB); DBG("\n");
  DBG("dC is: ");
  DBG_(dC); DBG("\n");


  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  // Swizzles for LDSM and 128b k-major loads
  auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                  Layout<Shape <_8,Shape <_8, _8>>,
                                         Stride<_8,Stride<_1,_64>>>{});
  DBG("swizzle_atom is: ");
  DBG_(swizzle_atom); DBG("\n");

  auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));
  auto sC = make_layout(make_shape(bM, bN));
  DBG("sA is: ");
  DBG_(sA); DBG("\n");
  DBG("sB is: ");
  DBG_(sB); DBG("\n");
  DBG("sC is: ");
  DBG_(sC); DBG("\n");

  // Define the thread layouts (static)

  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});               // Val layout  1x8 k-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});               // Val layout  1x8 n-major
  DBG("copyA is: ");
  DBG_(copyA); DBG("\n");
  DBG("copyB is: ");
  DBG_(copyB); DBG("\n");

  TiledMMA mmaC = make_tiled_mma(SM80_16x8x8_F16F16F16F16_TN{},
                                 Layout<Shape<_2,_2>>{},    // 2x2x1 MMA Atoms
                                 Tile<_32,_32,_16>{});      // 32x32x16 Tiled MMA for LDSM
  DBG("mmaC is: ");
  DBG_(mmaC); DBG("\n");

  //Copy_Atom<DefaultCopy, half_t> s2r_atom_A;
  //Copy_Atom<UniversalCopy<half_t>, half_t> s2r_atom_A;
  //Copy_Atom<SM75_U32x1_LDSM_N, half_t> s2r_atom_A;
  //Copy_Atom<SM75_U32x2_LDSM_N, half_t> s2r_atom_A;
  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;

  //Copy_Atom<DefaultCopy, half_t> s2r_atom_B;
  //Copy_Atom<UniversalCopy<half_t>, half_t> s2r_atom_B;
  //Copy_Atom<SM75_U32x1_LDSM_N, half_t> s2r_atom_B;
  //Copy_Atom<SM75_U32x2_LDSM_N, half_t> s2r_atom_B;
  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_B;

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  // print_latex(copyA);
  print_latex(copyB);
  // print_latex(mmaC);
#endif

  int smem_size = int(sizeof(SharedStorage<cute::half_t, cute::half_t, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));

  auto kernel_fptr = gemm_device<
    decltype(prob_shape), decltype(cta_tiler),
    cute::half_t, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_A),
    cute::half_t, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
    cute::half_t, decltype(dC), decltype(sC), decltype(mmaC),
    decltype(alpha), decltype(beta)>;

  // Set L1 to be SMEM only
  cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA, s2r_atom_A,
       B, dB, sB, copyB, s2r_atom_B,
       C, dC, sC, mmaC,
       alpha, beta);
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK, bP));             // (m,k,p) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK, bP));             // (n,k,p) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)

  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                    Layout<Shape<_32,_8>>{}, // Thr layout 32x8 m-major
                                    Layout<Shape< _4,_1>>{});// Val layout  4x1 m-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                    Layout<Shape<_32,_8>>{}, // Thr layout 32x8 n-major
                                    Layout<Shape< _4,_1>>{});// Val layout  4x1 n-major

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, smem_size, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA, AutoVectorizingCopy{},
       B, dB, sB, copyB, AutoVectorizingCopy{},
       C, dC, sC, mmaC,
       alpha, beta);
}

// Setup params for a TN GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA_atom                  = make_layout(make_shape (      bM,          bK),
                                              make_stride(Int<1>{}, bM+Int<1>{})); // (m,k) -> smem_idx; padded m-major
  [[maybe_unused]] auto sB_atom = make_layout(make_shape (      bN,          bK),
                                              make_stride(Int<1>{}, bN+Int<1>{})); // (n,k) -> smem_idx; padded n-major
  auto sA = tile_to_shape(sA_atom, make_shape(bM, bK, bP));
  auto sB = tile_to_shape(sA_atom, make_shape(bN, bK, bP));
  auto sC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx

  // Define the thread layouts (static)

  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TA>, TA>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TB>, TB>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, smem_size, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA, AutoVectorizingCopy{},
       B, dB, sB, copyB, AutoVectorizingCopy{},
       C, dC, sC, mmaC,
       alpha, beta);
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta,
     TC      * C, int ldC,
     cudaStream_t stream = 0)
{
  if (transA == 'N' && transB == 'T') {
    return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  } else
  if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
  }
  assert(false && "Not implemented");
}

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                  \
  {                                                                            \
    cutlass::Status error = status;                                            \
    if (error != cutlass::Status::kSuccess) {                                  \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error)      \
                << " at: " << __LINE__ << std::endl;                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                                     \
  {                                                                            \
    cudaError_t error = status;                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)        \
                << " at line: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU
 * stream
 */
struct GpuTimer {
  cudaStream_t _stream_id;
  cudaEvent_t _start;
  cudaEvent_t _stop;
  cutlass::gemm::GemmCoord problem_size = {0, 0, 0};

  /// Constructor
  GpuTimer() : _stream_id(0) {
    CUDA_CHECK(cudaEventCreate(&_start));
    CUDA_CHECK(cudaEventCreate(&_stop));
  }

  /// Destructor
  ~GpuTimer() {
    CUDA_CHECK(cudaEventDestroy(_start));
    CUDA_CHECK(cudaEventDestroy(_stop));
  }

  void set(cutlass::gemm::GemmCoord &_problem_size) {
    problem_size = _problem_size;
  }

  /// Start the timer for a given stream (defaults to the default stream)
  void start(cudaStream_t stream_id = 0) {
    _stream_id = stream_id;
    CUDA_CHECK(cudaEventRecord(_start, _stream_id));
  }

  /// Stop the timer
  void stop() { CUDA_CHECK(cudaEventRecord(_stop, _stream_id)); }

  /// Return the elapsed time (in milliseconds)
  float elapsed_millis() {
    float elapsed = 0.0;
    CUDA_CHECK(cudaEventSynchronize(_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
    return elapsed;
  }

  void bind_run(std::string name, const std::function<void()> &kernel,
                int test_time = 1) {
    float run_ms = 0;
    for (int i = 0; i < test_time; i++) {
      start();
      kernel();
      stop();
      run_ms += elapsed_millis();
    }
    run_ms /= (float)test_time;
    if (problem_size.product()) {
      double gflops = (double)problem_size.product() * 2 / 1e9 / (run_ms / 1e3);
      std::printf("[%20s] Runtime: %f(ms) Gflops: %f\n", name.c_str(), run_ms,
                  gflops);
    } else {
      std::printf("[%20s] Runtime: %f(ms)\n", name.c_str(), run_ms);
    }
  }

  template <typename E, typename L>
  void testEqual(std::string name, cutlass::HostTensor<E, L> &a,
                 cutlass::HostTensor<E, L> &b, bool ifprint = 0) {
    a.sync_host();
    b.sync_host();
    bool passed =
        cutlass::reference::host::TensorEquals(a.host_view(), b.host_view());

    if (passed)
      std::printf("[%20s] PASS\n", name.c_str());
    else {
      std::printf("[%20s] FAIL\n", name.c_str());
      if (ifprint) {
        std::cout << "[a]:\n" << a.host_view() << std::endl;
        std::cout << "[b]:\n" << b.host_view() << std::endl;
        cutlass::reference::host::TensorSub<E, L, E, L, E, L>(a.host_view(),
                                                              b.host_ref());
        std::cout << "diff:\n" << a.host_view() << std::endl;
      }
    }
  }

  template <typename E, typename L>
  void printTensor(std::string name, cutlass::HostTensor<E, L> &a) {
    a.sync_host();
    std::cout << name << ":\n" << a.host_view() << std::endl;
  }
};

// The code section below describes datatype for input, output matrices and
// computation between elements in input matrices.
using ElementAccumulator = cute::half_t; // <- data type of accumulator
using ElementComputeEpilogue =
    ElementAccumulator; // <- data type of epilogue operations
// using ElementInputA =
//     cutlass::bfloat16_t; // <- data type of elements in input matrix A
// using ElementInputB =
//     cutlass::bfloat16_t;     // <- data type of elements in input matrix B
using ElementInputA = cute::half_t;
using ElementInputB = cute::half_t;
using ElementOutput = cute::half_t; // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices.
// Column Major for Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

// This code section describes whether you want to use tensor cores or regular
// SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
// using ShapeMMAThreadBlock =
//     cutlass::gemm::GemmShape<128, 128, 16>; // <- threadblock tile M = 128, N =
//                                             // 128, K = 16
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 16>; // <- threadblock tile M = 128, N =

// This code section describes tile size a warp will compute
using ShapeMMAWarp =
    cutlass::gemm::GemmShape<64, 64, 16>; // <- warp tile M = 64, N = 64, K = 16
// This code section describes tile size a warp will compute
// using ShapeMMAWarp =
//     cutlass::gemm::GemmShape<32, 32, 16>; // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp =
    cutlass::gemm::GemmShape<16, 8, 8>; // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, // <- data type of output matrix
    128 / cutlass::sizeof_bits<
              ElementOutput>::value, // <- the number of elements per vectorized
                                     // memory access. For a byte, it's 16
                                     // elements. This becomes the vector width
                                     // of math instructions in the epilogue too
    ElementAccumulator,      // <- data type of accumulator
    ElementComputeEpilogue>; // <- data type for alpha/beta in linear
                             // combination function

// Number of pipelines you want to use
constexpr int NumStages = 3;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
    LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
    ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

struct MMAarguments {
  cutlass::gemm::GemmCoord problem_size;
  ElementInputA *A;
  ElementInputB *B;
  ElementAccumulator *C;
  ElementOutput *D;
};

template <typename T> __device__ void printvar(T var) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || blockIdx.y != 0)
    return;
  printf("%d\n", (int)var);
}

template <typename T>
__device__ void printtile(T *arr, int row, int col, bool rowMajor) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || blockIdx.y != 0)
    return;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (rowMajor)
        printf("%d ", (int)arr[i * col + j]);
      else
        printf("%d ", (int)arr[j * row + i]);
    }
    printf("\n");
  }
}

__device__ bool test(int a0, int b0, int a1, int b1) {
  return a0 < a1 && b0 < b1;
}


// Create a tuple of problem size for matrix multiplication
// cutlass::gemm::GemmCoord problem_size = {5120, 4096, 4096};
cutlass::gemm::GemmCoord problem_size = {4096, 4096, 4096};
// cutlass::gemm::GemmCoord problem_size = {128, 128, 16};
// cutlass::gemm::GemmCoord problem_size = {64, 64, 16};

// Initialize tensors using CUTLASS helper functions
cutlass::HostTensor<ElementInputA, LayoutInputA>
    tensor_a; // <- Create matrix A with dimensions M x K
cutlass::HostTensor<ElementInputB, LayoutInputB>
    tensor_b; // <- Create matrix B with dimensions K x N
cutlass::HostTensor<ElementOutput, LayoutOutput>
    tensor_c; // <- Create matrix C with dimensions M x N
cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d;
cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_low;

GpuTimer timer;

int main(int argc, char **argv) {
  //////////////////////////INIT////////////////////////////////
    if (argc >= 2)
        problem_size.m() = atoi(argv[1]);
    if (argc >= 3)
        problem_size.n() = atoi(argv[2]);
    if (argc >= 4)
        problem_size.k() = atoi(argv[3]);

    printf("[%20s] (%d,%d,%d)\n", "problem size", problem_size.m(),
            problem_size.n(), problem_size.k());

    tensor_a.resize(problem_size.mk());
    tensor_b.resize(problem_size.kn());
    tensor_c.resize(problem_size.mn());
    tensor_d.resize(problem_size.mn());
    tensor_d_low.resize(problem_size.mn());

    timer.set(problem_size);

    int m = problem_size.m();
    int k = problem_size.k();
    int n = problem_size.n();

    // My initialization of tensor_a, tensor_b and tensor_c
    // for (int i = 0; i < m*k; i++) {
    //     tensor_a.host_data()[i] = ElementInputA(i);
    // }

    // int stride_row = 1, stride_col = k;
    // for (int i = 0; i < k; i++) {
    //     for (int j = 0; j < n; j++) {
    //         tensor_b.host_data()[j * k + i] = ElementInputB(i * n + j);
    //     }
    // }

    // for (int i = 0; i < m*n; i++) {
    //     tensor_c.host_data()[i] = ElementAccumulator(0.0f);
    // }

    // tensor_a.sync_device();
    // tensor_b.sync_device();
    // tensor_c.sync_device();

    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < k; ++j) {
    //         if (i < 10 && j < 10) {
    //         DBG("a[%d, %d] = %f", i, j, tensor_a.at(MatrixCoord(i, j)));
    //         }
    //     }
    //     DBG("\n");
    // }
    
    // for (int i = 0; i < k; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         if (i < 10 && j < 10) {
    //         DBG("b[%d, %d] = %f", i, j, tensor_b.at(MatrixCoord(i, j)));
    //         }
    //     }
    //     DBG("\n");
    // }

    cutlass::reference::device::TensorFillRandomUniform(
        tensor_a.device_view(), 1, ElementInputA(4.f), ElementInputA(-4.f), 0);

    cutlass::reference::device::TensorFillRandomUniform(
        tensor_b.device_view(), 2, ElementInputB(4.f), ElementInputB(-4.f), 0);

    cutlass::reference::device::TensorFillRandomUniform(
        tensor_c.device_view(), 3, ElementAccumulator(1.f),
        ElementAccumulator(1.f), 0);
    tensor_d_low.copy_in_device_to_device(tensor_c.device_data());

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as
    // arguments to launch instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{
        problem_size,          // <- problem size of matrix multiplication
        tensor_a.device_ref(), // <- reference to matrix A on device
        tensor_b.device_ref(), // <- reference to matrix B on device
        tensor_c.device_ref(), // <- reference to matrix C on device
        tensor_d.device_ref(), // <- reference to matrix D on device
        {alpha, beta},         // <- tuple of alpha and beta
        split_k_slices};       // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix
    // multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);
    printf("ThreadBlockShape dimensions: M=%d, N=%d, K=%d\n", 
    Gemm::ThreadblockShape::kM, 
    Gemm::ThreadblockShape::kN, 
    Gemm::ThreadblockShape::kK);
    printf("WarpShape dimensions: M=%d, N=%d, K=%d\n", 
    Gemm::WarpShape::kM, 
    Gemm::WarpShape::kN, 
    Gemm::WarpShape::kK);
    printf("InstructionShape dimensions: M=%d, N=%d, K=%d\n", 
    Gemm::InstructionShape::kM, 
    Gemm::InstructionShape::kN, 
    Gemm::InstructionShape::kK);
    printf("Stages: %d\n", Gemm::kStages);


  ////////////////////cutlassMMA////////////////////////////////
#if ENABLE_CUTLASS
    timer.bind_run("cutlassMMA", [&] {
        // Launch CUTLASS kernel
        status = gemm_op();
    });
#endif
    CUTLASS_CHECK(status);

    {
        // my low level api implementation
        char transA = 'T';
        char transB = 'N';
        int m = problem_size.m();
        int n = problem_size.n();
        int k = problem_size.k();
        float alpha = 1.0f;
        float beta = 0.0f;

        int ldA = 0, ldB = 0, ldC = n;

        if (transA == 'N') {
            ldA = m;
        } else if (transA == 'T') {
            ldA = k;
        } else {
            assert(false);
        }

        if (transB == 'N') {
            ldB = k;
        } else if (transB == 'T') {
            ldB = n;
        } else {
            assert(false);
        }

        ElementInputA* d_A = tensor_a.device_data();
        ElementInputB* d_B = tensor_b.device_data();
        ElementAccumulator* d_C = tensor_d_low.device_data();

        // cudaMemcpy(d_A, tensor_a.device_data(), m*k*sizeof(ElementInputA), cudaMemcpyDeviceToDevice);
        // cudaMemcpy(d_B, tensor_b.device_data(), k*n*sizeof(ElementInputB), cudaMemcpyDeviceToDevice);
        // cudaMemcpy(d_C, tensor_c.device_data(), m*n*sizeof(ElementAccumulator), cudaMemcpyDeviceToDevice);
        
        timer.bind_run("my_gemm", [&] {
            gemm(transA, transB, m, n, k,
                alpha, 
                d_A, ldA,
                d_B, ldB,
                beta,
                d_C, ldC);
        });
        
        
        // tensor_d_low.copy_in_device_to_device(d_C);
        timer.testEqual<ElementOutput, LayoutOutput>("MMA_base==low_level", tensor_d,
                                                tensor_d_low, false);
        tensor_d_low.sync_host();
        tensor_d.sync_host();
        bool passed = true;
        for (int i = 0; i < m*n; i++) {
          if (static_cast<float>(tensor_d.host_data()[i]) - static_cast<float>(tensor_d_low.host_data()[i]) > 1e-6) {
            // DBG("d[%d] = %5.4f, d_my[%d] = %5.4f; ", i, static_cast<float>(tensor_d.host_data()[i]), i, static_cast<float>(tensor_d_low.host_data()[i]));
            passed = false;
            break;
          }
        }
        printf("[%20s] %s\n", "MMA_base==low_level", passed ? "PASS" : "FAIL");
        // for (int i = 0; i < 100; i++) {
        //     DBG("d_my[%d] = %5.2f, ", i, tensor_d_low.host_data()[i]);
        //     if (i > 0 && (i % 10 == 0)) DBG("\n");
        // }
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
  //////////////////////GEMM_MMA///////////////////////
//   {
// #if ENABLE_CUTLASS
//     cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_mma_d(
//         problem_size.mn());
//     cutlass::reference::device::TensorFillRandomUniform(
//         tensor_mma_d.device_view(), 3, ElementAccumulator(0.f),
//         ElementAccumulator(0.f), 0);
// #endif

//     timer.bind_run("MMA_base", [&] {
//       MMAarguments mmaArg {
//         problem_size, tensor_a.device_data(), tensor_b.device_data(),
//             tensor_c.device_data(),
// #if ENABLE_CUTLASS
//             tensor_mma_d.device_data()
// #else
//                 tensor_d.device_data()
// #endif
//       };
//       launch_GEMM_MMA(mmaArg);
//     });

// #if ENABLE_CUTLASS
//     timer.testEqual<ElementOutput, LayoutOutput>("MMA_base==ref", tensor_d,
//                                                  tensor_mma_d);
// #endif
//   }
}