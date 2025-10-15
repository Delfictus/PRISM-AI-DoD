// Tensor Core Matrix Multiplication using WMMA API
// Requires Compute Capability >= 7.0 (Volta+)
// Optimized for Ada Lovelace (Compute 12.0) - RTX 5070

#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// Tensor Core matmul: C = A * B
// A: m x k (FP16)
// B: k x n (FP16)
// C: m x n (FP32)
// Uses 16x16x16 WMMA tiles for Tensor Cores

extern "C" __global__ void tensor_core_matmul_wmma(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Warp and thread indices
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // WMMA dimensions: 16x16x16 (M x N x K)
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // Calculate the warp's position in output matrix
    int warp_row = warpM * WMMA_M;
    int warp_col = warpN * WMMA_N;

    // Bounds check
    if (warp_row >= M || warp_col >= N) return;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize the output to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over K dimension in WMMA_K chunks
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warp_row;
        int aCol = k;
        int bRow = k;
        int bCol = warp_col;

        // Bounds check for partial tiles
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store the output
    int cRow = warp_row;
    int cCol = warp_col;

    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
    }
}

// Optimized version with shared memory for larger matrices
extern "C" __global__ void tensor_core_matmul_wmma_shared(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for tile caching
    __shared__ half As[128][128];  // Tile of A
    __shared__ half Bs[128][128];  // Tile of B

    // Warp indices
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    int warp_row = warpM * WMMA_M;
    int warp_col = warpN * WMMA_N;

    if (warp_row >= M || warp_col >= N) return;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Tile over K dimension
    const int TILE_K = 128;
    for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {
        // Cooperatively load tiles into shared memory
        // Each thread loads multiple elements
        int load_idx = threadIdx.x + threadIdx.y * blockDim.x;
        int num_threads = blockDim.x * blockDim.y;

        // Load A tile
        for (int i = load_idx; i < TILE_K * WMMA_M; i += num_threads) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = warp_row + row;
            int global_col = tile_k + col;

            if (global_row < M && global_col < K) {
                As[row][col] = A[global_row * K + global_col];
            } else {
                As[row][col] = __float2half(0.0f);
            }
        }

        // Load B tile
        for (int i = load_idx; i < TILE_K * WMMA_N; i += num_threads) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;
            int global_row = tile_k + row;
            int global_col = warp_col + col;

            if (global_row < K && global_col < N) {
                Bs[row][col] = B[global_row * N + global_col];
            } else {
                Bs[row][col] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Compute with WMMA using shared memory tiles
        for (int k = 0; k < TILE_K; k += WMMA_K) {
            wmma::load_matrix_sync(a_frag, &As[0][k], TILE_K);
            wmma::load_matrix_sync(b_frag, &Bs[k][0], WMMA_N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        __syncthreads();
    }

    // Store result
    if (warp_row < M && warp_col < N) {
        wmma::store_matrix_sync(C + warp_row * N + warp_col, acc_frag, N, wmma::mem_row_major);
    }
}
