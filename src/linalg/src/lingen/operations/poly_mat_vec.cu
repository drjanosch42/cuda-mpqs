// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "lingen/operations/poly_mat_vec.h"
#include "vec_type.h" 
#include "hpc_logger.h"
#include <vector>
#include <cstring>
#include <algorithm>

namespace lingen {

namespace kernel {

__device__ __forceinline__ uint32_t warp_transpose32_shfl(uint32_t val) {
    uint32_t tmp;
    int lane = threadIdx.x & 31;
    tmp = __shfl_xor_sync(0xFFFFFFFF, val, 16);
    if (lane & 16) val = (val & 0xFFFF0000) | (tmp >> 16);
    else           val = (val & 0x0000FFFF) | (tmp << 16);
    tmp = __shfl_xor_sync(0xFFFFFFFF, val, 8);
    if (lane & 8) val = (val & 0xFF00FF00) | ((tmp >> 8) & 0x00FF00FF);
    else          val = (val & 0x00FF00FF) | ((tmp << 8) & 0xFF00FF00);
    tmp = __shfl_xor_sync(0xFFFFFFFF, val, 4);
    if (lane & 4) val = (val & 0xF0F0F0F0) | ((tmp >> 4) & 0x0F0F0F0F);
    else          val = (val & 0x0F0F0F0F) | ((tmp << 4) & 0xF0F0F0F0);
    tmp = __shfl_xor_sync(0xFFFFFFFF, val, 2);
    if (lane & 2) val = (val & 0xCCCCCCCC) | ((tmp >> 2) & 0x33333333);
    else          val = (val & 0x33333333) | ((tmp << 2) & 0xCCCCCCCC);
    tmp = __shfl_xor_sync(0xFFFFFFFF, val, 1);
    if (lane & 1) val = (val & 0xAAAAAAAA) | ((tmp >> 1) & 0x55555555);
    else          val = (val & 0x55555555) | ((tmp << 1) & 0xAAAAAAAA);
    return val;
}

template<int M>
__global__ void k_transpose_poly_gf2(const uint64_t* __restrict__ Pi_u64, uint64_t* __restrict__ PiT_u64, int Pi_len) {
    constexpr int SEG = M / 128;
    constexpr size_t MATSTRIDEWORDS = size_t(M) * size_t(M) / 64;
    int k = (int)blockIdx.z;
    if (k >= Pi_len) return;
    const uint4* A4 = reinterpret_cast<const uint4*>(Pi_u64 + size_t(k) * MATSTRIDEWORDS);
    uint4* AT4      = reinterpret_cast<uint4*>(PiT_u64 + size_t(k) * MATSTRIDEWORDS);
    const int tr = (int)blockIdx.y; 
    const int tc = (int)blockIdx.x; 
    const int lane = (int)threadIdx.x;
    const int warp = (int)threadIdx.y; 
    const int in_r = warp * 32 + lane; 
    const int Arow = tr * 128 + in_r;
    const uint4 in = A4[Arow * SEG + tc];
    uint32_t tx = warp_transpose32_shfl(in.x);
    uint32_t ty = warp_transpose32_shfl(in.y);
    uint32_t tz = warp_transpose32_shfl(in.z);
    uint32_t tw = warp_transpose32_shfl(in.w);
    __shared__ uint32_t sOut[128][4];
    sOut[0 * 32 + lane][warp] = tx;
    sOut[1 * 32 + lane][warp] = ty;
    sOut[2 * 32 + lane][warp] = tz;
    sOut[3 * 32 + lane][warp] = tw;
    __syncthreads();
    const int out_r = in_r;
    const int ATrow = tc * 128 + out_r;
    uint4 out;
    out.x = sOut[out_r][0]; out.y = sOut[out_r][1]; out.z = sOut[out_r][2]; out.w = sOut[out_r][3];
    AT4[ATrow * SEG + tr] = out;
}

template<int M, int G>
__device__ __forceinline__ typename VecType<M>::Type mask_low_G_bits(typename VecType<M>::Type v) {
    using Vec = VecType<M>;
    using T   = typename Vec::Type;
    constexpr int WORDS_M = M / 64;
    constexpr int WORDS_G = G / 64;
    union RegView { T vec; uint64_t u64[WORDS_M]; };
    RegView r; r.vec = v;
    #pragma unroll
    for (int w = WORDS_G; w < WORDS_M; ++w) r.u64[w] = 0ULL;
    return r.vec;
}

// Computes W = V * Pi (Right Apply, but math Left Mul: v_row * M)
template<int M, int G>
__global__ void k_apply_pi_right_padded(const uint64_t* __restrict__ Pi_u64, int Pi_len, const typename VecType<M>::Type* __restrict__ V, int Lin, typename VecType<M>::Type* __restrict__ W, int Lout) {
    using Vec = VecType<M>;
    using T   = typename Vec::Type;
    constexpr int WORDS_G = G / 64;
    constexpr int BITSPER = 64;
    extern __shared__ __align__(16) char pi_smem[];
    T* PiRows = reinterpret_cast<T*>(pi_smem); 
    const int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    T acc = Vec::zero();
    for (int k = 0; k < Pi_len; ++k) {
        const uint64_t* Pik_u64 = Pi_u64 + size_t(k) * (size_t(M) * size_t(M) / 64);
        const T* Pik_rows = reinterpret_cast<const T*>(Pik_u64);
        for (int r = (int)threadIdx.x; r < G; r += (int)blockDim.x) { PiRows[r] = Vec::load(Pik_rows + r); }
        __syncthreads();
        if (t < Lout) {
            const int ti = t - k;
            if (ti >= 0 && ti < Lin) {
                T v = Vec::load(V + ti);
                v = mask_low_G_bits<M, G>(v);
                union RegView { T vec; uint64_t u64[M/64]; } vv; vv.vec = v;
                #pragma unroll
                for (int limb = 0; limb < WORDS_G; ++limb) {
                    const uint64_t word = vv.u64[limb];
                    const int base = limb * BITSPER;
                    #pragma unroll
                    for (int bit = 0; bit < BITSPER; ++bit) {
                        if ((word >> bit) & 1ULL) acc = Vec::xor_val(acc, PiRows[base + bit]);
                    }
                }
            }
        }
        __syncthreads();
    }
    if (t < Lout) {
        acc = mask_low_G_bits<M, G>(acc);
        Vec::store(W + t, acc);
    }
}

template<int G>
__global__ void k_apply_pi_right_direct(const uint64_t* __restrict__ Pi_u64, int Pi_len, const typename VecType<G>::Type* __restrict__ V, int Lin, typename VecType<G>::Type* __restrict__ W, int Lout) {
    using Vec = VecType<G>;
    using T   = typename Vec::Type;
    constexpr int NUMLIMBS = G / 64;
    extern __shared__ __align__(16) char pi_smem[];
    T* PiRows = reinterpret_cast<T*>(pi_smem);
    int t = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    T acc = Vec::zero();
    for (int k = 0; k < Pi_len; k++) {
        const uint64_t* Pik_u64 = Pi_u64 + size_t(k) * (size_t(G) * size_t(G) / 64);
        const T* Pik_rows = reinterpret_cast<const T*>(Pik_u64);
        for (int r = int(threadIdx.x); r < G; r += int(blockDim.x)) { PiRows[r] = Vec::load(Pik_rows + r); }
        __syncthreads();
        if (t < Lout) {
            int ti = t - k;
            if (ti >= 0 && ti < Lin) {
                T v = Vec::load(V + ti);
                union RegView { T vec; uint64_t u64[NUMLIMBS]; } vv; vv.vec = v;
                #pragma unroll
                for (int limb = 0; limb < NUMLIMBS; limb++) {
                    uint64_t word = vv.u64[limb];
                    int base = limb * 64;
                    #pragma unroll
                    for (int bit = 0; bit < 64; bit++) {
                        if ((word >> bit) & 1ULL) acc = Vec::xor_val(acc, PiRows[base + bit]);
                    }
                }
            }
        }
        __syncthreads();
    }
    if (t < Lout) Vec::store(W + t, acc);
}

// Computes W = Pi * V (Left Apply, Matrix * Col Vector)
template<int G>
__global__ void k_apply_pi_left_direct(const uint64_t* __restrict__ Pi_u64, int Pi_len, const typename VecType<G>::Type* __restrict__ V, int Lin, typename VecType<G>::Type* __restrict__ W, int Lout) {
    using Vec = VecType<G>;
    using T   = typename Vec::Type;
    constexpr int WORDS_G = G / 64;
    extern __shared__ __align__(16) char pi_smem[];
    T* PiRows = reinterpret_cast<T*>(pi_smem); 
    const int t = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    uint64_t acc_words[WORDS_G];
    #pragma unroll
    for (int w = 0; w < WORDS_G; ++w) acc_words[w] = 0ULL;
    for (int k = 0; k < Pi_len; ++k) {
        const uint64_t* Pik_u64 = Pi_u64 + size_t(k) * (size_t(G) * size_t(G) / 64);
        for (int r = int(threadIdx.x); r < G; r += int(blockDim.x)) { PiRows[r] = Vec::load(reinterpret_cast<const T*>(Pik_u64) + r); }
        __syncthreads();
        if (t < Lout) {
            const int ti = t - k;
            if (ti >= 0 && ti < Lin) {
                T v = Vec::load(V + ti);
                union VView { T vec; uint64_t u64[WORDS_G]; } vv; vv.vec = v;
                #pragma unroll
                for (int outw = 0; outw < WORDS_G; ++outw) {
                    uint64_t out_word = 0ULL;
                    #pragma unroll
                    for (int i = 0; i < 64; ++i) {
                        int row_idx = outw * 64 + i;
                        union RView { T vec; uint64_t u64[WORDS_G]; } rowv; rowv.vec = PiRows[row_idx];
                        int p = 0;
                        #pragma unroll
                        for (int w = 0; w < WORDS_G; ++w) p ^= __popcll(rowv.u64[w] & vv.u64[w]);
                        if (p & 1) out_word |= (1ULL << i);
                    }
                    acc_words[outw] ^= out_word;
                }
            }
        }
        __syncthreads();
    }
    if (t < Lout) {
        union OView { T vec; uint64_t u64[WORDS_G]; } outv;
        #pragma unroll
        for (int w = 0; w < WORDS_G; ++w) outv.u64[w] = acc_words[w];
        Vec::store(W + t, outv.vec);
    }
}
} // namespace kernel

namespace ref {
static inline bool get_bit(const uint64_t* mat, int M, int r, int c) {
    int words_per_row = M / 64;
    return (mat[r * words_per_row + (c / 64)] >> (c % 64)) & 1ULL;
}
static inline bool get_vec_bit(const uint64_t* V, int M, int t, int bit_idx) {
    int words_per_vec = M / 64;
    const uint64_t* vec = V + t * words_per_vec;
    return (vec[bit_idx / 64] >> (bit_idx % 64)) & 1ULL;
}
static inline void xor_vec_bit(uint64_t* W, int M, int t, int bit_idx) {
    int words_per_vec = M / 64;
    uint64_t* vec = W + t * words_per_vec;
    vec[bit_idx / 64] ^= (1ULL << (bit_idx % 64));
}

// CPU Right Apply: W = V * Pi (Vector * Matrix)
template<int M, int G>
void cpu_apply_pi_right(const PolyMatrixView<M>& Pi, const uint64_t* V, int Lin, uint64_t* W, int Lout) {
    size_t pi_bytes = Pi.length * PolyMatrixView<M>::MAT_SIZE_BYTES;
    size_t v_bytes  = (size_t)Lin * (M/8);
    size_t w_bytes  = (size_t)Lout * (M/8);
    std::vector<uint64_t> hPi(pi_bytes / 8);
    std::vector<uint64_t> hV(v_bytes / 8);
    std::vector<uint64_t> hW(w_bytes / 8, 0);
    cudaMemcpy(hPi.data(), Pi.data, pi_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hV.data(), V, v_bytes, cudaMemcpyDeviceToHost);

    for (int t = 0; t < Lout; ++t) {
        int k_max = std::min(t, (int)Pi.length - 1);
        for (int k = 0; k <= k_max; ++k) {
            int ti = t - k;
            if (ti >= Lin) continue;
            const uint64_t* Pi_k = hPi.data() + k * PolyMatrixView<M>::MAT_STRIDE_WORDS;
            for (int c = 0; c < G; ++c) { 
                bool dot = false;
                for (int r = 0; r < G; ++r) { 
                    // V[r] * Pi[r, c]
                    if (get_vec_bit(hV.data(), M, ti, r) && get_bit(Pi_k, M, r, c)) dot = !dot;
                }
                if (dot) xor_vec_bit(hW.data(), M, t, c);
            }
        }
    }
    cudaMemcpy(W, hW.data(), w_bytes, cudaMemcpyHostToDevice);
}

// CPU Left Apply: W = Pi * V (Matrix * Vector)
template<int M, int G>
void cpu_apply_pi_left(const PolyMatrixView<M>& Pi, const uint64_t* V, int Lin, uint64_t* W, int Lout) {
    size_t pi_bytes = Pi.length * PolyMatrixView<M>::MAT_SIZE_BYTES;
    size_t v_bytes  = (size_t)Lin * (M/8);
    size_t w_bytes  = (size_t)Lout * (M/8);
    std::vector<uint64_t> hPi(pi_bytes / 8);
    std::vector<uint64_t> hV(v_bytes / 8);
    std::vector<uint64_t> hW(w_bytes / 8, 0);
    cudaMemcpy(hPi.data(), Pi.data, pi_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hV.data(), V, v_bytes, cudaMemcpyDeviceToHost);

    for (int t = 0; t < Lout; ++t) {
        int k_max = std::min(t, (int)Pi.length - 1);
        for (int k = 0; k <= k_max; ++k) {
            int ti = t - k;
            if (ti >= Lin) continue;
            const uint64_t* Pi_k = hPi.data() + k * PolyMatrixView<M>::MAT_STRIDE_WORDS;
            for (int r = 0; r < G; ++r) { 
                bool dot = false;
                for (int c = 0; c < G; ++c) {
                    // Pi[r, c] * V[c]
                    if (get_bit(Pi_k, M, r, c) && get_vec_bit(hV.data(), M, ti, c)) dot = !dot;
                }
                if (dot) xor_vec_bit(hW.data(), M, t, r);
            }
        }
    }
    cudaMemcpy(W, hW.data(), w_bytes, cudaMemcpyHostToDevice);
}
} // namespace ref

template<int M, int G>
void PolyMatVec::apply_right(const PolyMatrixView<M>& Pi_dev, const uint64_t* V_dev, int Lin, uint64_t* W_dev, int Lout, PolyMatVecBackend backend, cudaStream_t stream) {
    if (backend == PolyMatVecBackend::CPU_Reference) {
        cudaStreamSynchronize(stream);
        ref::cpu_apply_pi_right<M, G>(Pi_dev, V_dev, Lin, W_dev, Lout);
        return;
    }
    if (M == G) {
        using VecG = typename VecType<G>::Type;
        size_t shmem = G * sizeof(VecG);
        int threads = 256;
        int blocks = (Lout + threads - 1) / threads;
        kernel::k_apply_pi_right_direct<G><<<blocks, threads, shmem, stream>>>(Pi_dev.data, (int)Pi_dev.length, reinterpret_cast<const VecG*>(V_dev), Lin, reinterpret_cast<VecG*>(W_dev), Lout);
    } else {
        using VecT = typename VecType<M>::Type;
        size_t shmem = G * sizeof(VecT);
        int threads = 256;
        int blocks = (Lout + threads - 1) / threads;
        kernel::k_apply_pi_right_padded<M, G><<<blocks, threads, shmem, stream>>>(Pi_dev.data, (int)Pi_dev.length, reinterpret_cast<const VecT*>(V_dev), Lin, reinterpret_cast<VecT*>(W_dev), Lout);
    }
}

template<int M, int G>
void PolyMatVec::apply_left(const PolyMatrixView<M>& Pi_dev, const uint64_t* V_dev, int Lin, uint64_t* W_dev, int Lout, PolyMatVecBackend backend, cudaStream_t stream) {
    if (backend == PolyMatVecBackend::CPU_Reference) {
        cudaStreamSynchronize(stream);
        ref::cpu_apply_pi_left<M, G>(Pi_dev, V_dev, Lin, W_dev, Lout);
        return;
    }
    if (M == G) {
        using VecG = typename VecType<G>::Type;
        size_t shmem = G * sizeof(VecG);
        int threads = 256;
        int blocks = (Lout + threads - 1) / threads;
        kernel::k_apply_pi_left_direct<G><<<blocks, threads, shmem, stream>>>(Pi_dev.data, (int)Pi_dev.length, reinterpret_cast<const VecG*>(V_dev), Lin, reinterpret_cast<VecG*>(W_dev), Lout);
    } else {
        // Fallback for padded M > G left application via Transpose + Right
        uint64_t* d_PiT_ptr;
        size_t pi_bytes = Pi_dev.length * PolyMatrixView<M>::MAT_SIZE_BYTES;
        cudaMalloc(&d_PiT_ptr, pi_bytes);
        PolyMatrixView<M> PiT(d_PiT_ptr, Pi_dev.length);
        transpose<M>(PiT, Pi_dev, stream);
        apply_right<M, G>(PiT, V_dev, Lin, W_dev, Lout, backend, stream);
        cudaFree(d_PiT_ptr);
    }
}

template<int M>
void PolyMatVec::transpose(PolyMatrixView<M> dst, PolyMatrixView<M> src, cudaStream_t stream) {
    int len = (dst.length < src.length) ? dst.length : src.length;
    dim3 grid(M / 128, M / 128, len);
    dim3 block(32, 4, 1);
    kernel::k_transpose_poly_gf2<M><<<grid, block, 0, stream>>>(src.data, dst.data, len);
}

// Instantiations
template void PolyMatVec::apply_right<64, 64>(const PolyMatrixView<64>&, const uint64_t*, int, uint64_t*, int, PolyMatVecBackend, cudaStream_t);
template void PolyMatVec::apply_left<64, 64>(const PolyMatrixView<64>&, const uint64_t*, int, uint64_t*, int, PolyMatVecBackend, cudaStream_t);
template void PolyMatVec::apply_right<128, 128>(const PolyMatrixView<128>&, const uint64_t*, int, uint64_t*, int, PolyMatVecBackend, cudaStream_t);
template void PolyMatVec::apply_left<128, 128>(const PolyMatrixView<128>&, const uint64_t*, int, uint64_t*, int, PolyMatVecBackend, cudaStream_t);
template void PolyMatVec::apply_right<256, 256>(const PolyMatrixView<256>&, const uint64_t*, int, uint64_t*, int, PolyMatVecBackend, cudaStream_t);
template void PolyMatVec::apply_left<256, 256>(const PolyMatrixView<256>&, const uint64_t*, int, uint64_t*, int, PolyMatVecBackend, cudaStream_t);
template void PolyMatVec::apply_right<512, 512>(const PolyMatrixView<512>&, const uint64_t*, int, uint64_t*, int, PolyMatVecBackend, cudaStream_t);
template void PolyMatVec::apply_left<512, 512>(const PolyMatrixView<512>&, const uint64_t*, int, uint64_t*, int, PolyMatVecBackend, cudaStream_t);

template void PolyMatVec::apply_right<128, 64>(const PolyMatrixView<128>&, const uint64_t*, int, uint64_t*, int, PolyMatVecBackend, cudaStream_t);
template void PolyMatVec::apply_left<128, 64>(const PolyMatrixView<128>&, const uint64_t*, int, uint64_t*, int, PolyMatVecBackend, cudaStream_t);
template void PolyMatVec::apply_right<256, 128>(const PolyMatrixView<256>&, const uint64_t*, int, uint64_t*, int, PolyMatVecBackend, cudaStream_t);
template void PolyMatVec::apply_left<256, 128>(const PolyMatrixView<256>&, const uint64_t*, int, uint64_t*, int, PolyMatVecBackend, cudaStream_t);
template void PolyMatVec::apply_right<512, 256>(const PolyMatrixView<512>&, const uint64_t*, int, uint64_t*, int, PolyMatVecBackend, cudaStream_t);
template void PolyMatVec::apply_left<512, 256>(const PolyMatrixView<512>&, const uint64_t*, int, uint64_t*, int, PolyMatVecBackend, cudaStream_t);

template void PolyMatVec::transpose<64>(PolyMatrixView<64>, PolyMatrixView<64>, cudaStream_t);
template void PolyMatVec::transpose<128>(PolyMatrixView<128>, PolyMatrixView<128>, cudaStream_t);
template void PolyMatVec::transpose<256>(PolyMatrixView<256>, PolyMatrixView<256>, cudaStream_t);
template void PolyMatVec::transpose<512>(PolyMatrixView<512>, PolyMatrixView<512>, cudaStream_t);

} // namespace lingen
