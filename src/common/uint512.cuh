// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <iomanip>

namespace mpqs {

// =============================================================================
// CUDA Intrinsics Helpers
// =============================================================================

__host__ __device__ __forceinline__ int clz32(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __clz(x);
#else
    return x ? __builtin_clz(x) : 32;
#endif
}

__host__ __device__ __forceinline__ int ctz32(uint32_t x) {
#ifdef __CUDA_ARCH__
    if (!x) return 32;  // __ffs(0) returns 0 → would yield -1 without guard
    return __ffs(x) - 1;
#else
    return x ? __builtin_ctz(x) : 32;
#endif
}

// =============================================================================
// Class Definition
// =============================================================================

/**
 * @brief A 512-bit Unsigned Integer class optimized for CUDA HPC.
 * 
 * Layout: Little-endian (limbs[0] is LSB).
 * Range: [0, 2^512 - 1].
 * Overflow/Underflow: Wraps silently (standard C unsigned behavior).
 */
class uint512 {
public:
    uint32_t limbs[16];

    // =========================================================================
    // Constructors
    // =========================================================================

    __host__ __device__ uint512() {
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 0; i < 16; i++) limbs[i] = 0;
    }

    __host__ __device__ uint512(uint32_t x) {
        limbs[0] = x;
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 1; i < 16; i++) limbs[i] = 0;
    }

    __host__ __device__ uint512(uint64_t x) {
        limbs[0] = (uint32_t)x;
        limbs[1] = (uint32_t)(x >> 32);
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 2; i < 16; i++) limbs[i] = 0;
    }

    __host__ __device__ uint512(unsigned __int128 x) {
        limbs[0] = (uint32_t)x;
        limbs[1] = (uint32_t)(x >> 32);
        limbs[2] = (uint32_t)(x >> 64);
        limbs[3] = (uint32_t)(x >> 96);
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 4; i < 16; i++) limbs[i] = 0;
    }

    __host__ __device__ uint512(const uint32_t (&in_limbs)[16]) {
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 0; i < 16; i++) limbs[i] = in_limbs[i];
    }

    // Create uint512 from decimal string
    // Simple implementation: result = result * 10 + digit.
    // Returns max_value 2^512 - 1 if parsing fails.
    __host__ __device__ uint512(const char* str) {
        mpqs::uint512 res((uint32_t)0);
	while (*str) {
	    if (*str >= '0' && *str <= '9') {
	        res.mult_uint32((uint32_t)10);
		int32_t digit = (*str - '0');
		if((0 < digit) && (digit < 10)) {
		    res.add_uint32((uint32_t)digit);
		} else if (digit != 0) {
		    // LOG(LOG_ERROR_CRITICAL) << "[uint512] ERROR: Non-decimal character encountert during str -> uint512 conversion";
		    // This allows subsequent code to catch the parsing error
		    *this = max_value(); 
		}
	    }
	    str++;
	}
	*this = res;
    }

    // =========================================================================
    // Factory Helpers
    // =========================================================================

    __host__ __device__ static uint512 max_value() {
        uint512 r;
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for(int i=0; i<16; i++) r.limbs[i] = 0xFFFFFFFF;
        return r;
    }

    // =========================================================================
    // Basic Arithmetic (In-Place)
    // =========================================================================

    __host__ __device__ void add(const uint512& other) {
        uint64_t carry = 0;
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 0; i < 16; i++) {
            uint64_t res = (uint64_t)limbs[i] + other.limbs[i] + carry;
            limbs[i] = (uint32_t)res;
            carry = res >> 32;
        }
    }

    __host__ __device__ void add_uint32(const uint32_t v) {
	uint64_t res = (uint64_t)limbs[0] + v;
	uint64_t carry = res >> 32;
	limbs[0] = (uint32_t)res;
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 1; carry & (i < 16); i++) {
	    res = (uint64_t)limbs[i] + carry;
            limbs[i] = (uint32_t)res;
            carry = res >> 32;
        }
    }

    // Optimization for sub (pure unsigned logic)
    __host__ __device__ void sub(const uint512& other) {
        uint32_t borrow = 0;
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
	for (int i = 0; i < 16; i++) {
	    uint32_t a = limbs[i];
	    uint32_t b = other.limbs[i];
	    uint32_t diff = a - b - borrow;
        
	    // Borrow happens if (a < b) OR (a == b AND borrow was 1)
	    // Equivalent to: borrow = (a < b + borrow) ? ... but be careful of overflow in (b+borrow)
	    // Safe check:
	    borrow = (a < b) || (borrow && a == b); 
        
	    limbs[i] = diff;
	}
    }

    /**
     * @brief O(N^2) Schoolbook Multiplication.
     * Truncates to lower 512 bits.
     */
    __host__ __device__ void mult(const uint512& other) {
        uint32_t res[16] = {0};
        
        // Loop unrolling hint for compiler, though outer loop is large
        for (int i = 0; i < 16; i++) {
            if (limbs[i] == 0) continue; // Sparse optimization
            uint64_t carry = 0;
            for (int j = 0; j < 16 - i; j++) {
                uint64_t prod = (uint64_t)limbs[i] * other.limbs[j] + res[i + j] + carry;
                res[i + j] = (uint32_t)prod;
                carry = prod >> 32;
            }
        }
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 0; i < 16; i++) limbs[i] = res[i];
    }

    /**
     * @brief In-place multiply by a 32-bit integer.
     * Much faster than full uint512 multiplication.
     */
    __host__ __device__ void mult_uint32(uint32_t v) {
        if (v == 0) { *this = uint512((uint32_t)0); return; }
        if (v == 1) return;
        uint64_t carry = 0;
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 0; i < 16; i++) {
            uint64_t prod = (uint64_t)limbs[i] * v + carry;
            limbs[i] = (uint32_t)prod;
            carry = prod >> 32;
        }
    }

   /**
     * @brief Non-destructive division by u32. Returns quotient.
     * Does NOT modify *this.
     * 
     * Used in polynomial generation to compute (a / p) without destroying 
     * the shared 'a' which other threads are reading.
     */
    __host__ __device__ uint512 div_uint32_const(uint32_t divisor) const {
        // Constructor initializes limbs to zero
        uint512 quotient; 
        
        if (divisor == 0) return quotient; // Handle div by zero gracefully
        
        uint64_t rem = 0;
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 15; i >= 0; i--) {
            uint64_t cur = (rem << 32) | limbs[i];
            quotient.limbs[i] = (uint32_t)(cur / divisor);
            rem = cur % divisor;
        }
        return quotient;
    }    

    // =========================================================================
    // Division & Modulo
    // =========================================================================

    /**
     * @brief Core Division Kernel.
     * Computes Quotient (*this) and Remainder (optional).
     * 
     * Correctly handles the 512-bit boundary using carry tracking.
     */
    __host__ __device__ void div_mod_core(const uint512& divisor, uint512* remainder_out) {
        if (divisor.is_zero()) return; // No-op / Undefined

        // Optimization: Dividend < Divisor
        if (*this < divisor) {
            if (remainder_out) *remainder_out = *this;
            *this = uint512((uint32_t)0);
            return;
        }

        uint512 dividend = *this;
        uint512 quotient((uint32_t)0);
        uint512 remainder((uint32_t)0);

        // Scan from MSB of dividend
        int bit_idx = dividend.msb();
        
        for (int i = bit_idx; i >= 0; i--) {
            // 1. Shift Remainder Left by 1 (R = R << 1)
            //    Must capture the bit shifted out of pos 511 (overflow_bit)
            uint32_t carry = 0;
            #if defined(__NVCC__) && defined(__CUDA_ARCH__)
                #pragma unroll
            #endif      
            for (int j = 0; j < 16; j++) {
                uint32_t val = remainder.limbs[j];
                uint32_t next_carry = val >> 31;
                remainder.limbs[j] = (val << 1) | carry;
                carry = next_carry;
            }
            uint32_t overflow_bit = carry;

            // 2. Inject next bit from Dividend
            uint32_t dividend_bit = (dividend.limbs[i / 32] >> (i % 32)) & 1;
            remainder.limbs[0] |= dividend_bit;

            // 3. Check if R >= D
            bool geq = false;
            if (overflow_bit) {
                // If remainder has 513 bits (bit 512 set), it's definitely >= 512-bit divisor
                geq = true;
            } else {
                // Standard comparison
                // Inline logic to avoid function call overhead in tight loop
                geq = true;
                #if defined(__NVCC__) && defined(__CUDA_ARCH__)
                    #pragma unroll
                #endif      
                for (int j = 15; j >= 0; j--) {
                    if (remainder.limbs[j] < divisor.limbs[j]) { geq = false; break; }
                    if (remainder.limbs[j] > divisor.limbs[j]) { break; }
                }
            }

            // 4. Update
            if (geq) {
                remainder.sub(divisor);
                quotient.limbs[i / 32] |= (1U << (i % 32));
            }
        }

        *this = quotient;
        if (remainder_out) *remainder_out = remainder;
    }

    __host__ __device__ void div(const uint512& divisor) {
        div_mod_core(divisor, nullptr);
    }

    __host__ __device__ void mod(const uint512& divisor) {
        uint512 rem;
        // Optimization: div_mod_core destroys *this (turns it into quotient).
        // To implement mod efficiently, we need a copy of *this.
        // There is no faster way than running division logic.
        uint512 temp = *this;
        temp.div_mod_core(divisor, &rem);
        *this = rem;
    }

    /**
     * @brief Computes (this + other) % N in-place.
     * Assumes this < N and other < N.
     * Relies on wrapping addition / subtraction.
     */
    __host__ __device__ void add_mod(const uint512& other, const uint512& N) {
        uint512 old_val = *this;  // Save state to detect wrap
	this->add(other);

	// Overflow logic:
	// 1. (*this < old_val): The 512-bit add wrapped around (standard overflow).
	// 2. (*this >= N): No wrap, but the result is larger than the modulus.
	if (*this < old_val || *this >= N) {
	    this->sub(N);
	}
    }
    
    /**
     * @brief Computes (this - other) % N in-place.
     * Assumes this < N and other < N.
     * Ensures the result is the unique representative in [0, N-1].
     * Relies on wrapping addition / subtraction.
     */
    __host__ __device__ void sub_mod(const uint512& other, const uint512& N) {
        if (*this < other) {
	    // To avoid overflow when adding N, we subtract 'other' 
            // and then add N to the wrapped result.
            this->sub(other);
	    this->add(N);
	} else {
	    this->sub(other);
	}
    }

    /**
     * @brief Computes (2 * this) % N in-place.
     * Assumes this < N. Equivalent to add_mod(*this, N).
     * Relies on wrapping addition / subtraction.
     */
    __host__ __device__ void double_mod(const uint512& N) {
        // Capture the bit that will be shifted out
        uint32_t carry = limbs[15] >> 31;
	this->lshift(1);
    
	if (carry || *this >= N) {
	    this->sub(N);
	}
    }
    
    /**
     * @brief Computes (*this * other) % modulus without intermediate overflow.
     * Uses 1024-bit intermediate storage.
     */
    __host__ __device__ void mul_mod(const uint512& other, const uint512& modulus) {
        if (modulus.is_zero()) return;

        // 1. Compute 1024-bit product
        uint32_t p[32];
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for(int i=0; i<32; i++) p[i] = 0;

        for (int i = 0; i < 16; i++) {
            if (limbs[i] == 0) continue;
            uint64_t carry = 0;
            for (int j = 0; j < 16; j++) {
                uint64_t term = (uint64_t)limbs[i] * other.limbs[j] + p[i+j] + carry;
                p[i+j] = (uint32_t)term;
                carry = term >> 32;
            }
            p[i+16] += (uint32_t)carry;
        }

        // 2. Reduce 1024 -> 512 bits
        uint512 remainder((uint32_t)0);
        
        // Find effective MSB of product to reduce loop count
        int p_msb = 1023;
        while(p_msb >= 0 && ((p[p_msb/32] >> (p_msb%32)) & 1) == 0) p_msb--;

        for (int i = p_msb; i >= 0; i--) {
            // Shift remainder << 1
            uint32_t carry = 0;
            #if defined(__NVCC__) && defined(__CUDA_ARCH__)
                #pragma unroll
            #endif      
            for (int j = 0; j < 16; j++) {
                uint32_t nxt = remainder.limbs[j] >> 31;
                remainder.limbs[j] = (remainder.limbs[j] << 1) | carry;
                carry = nxt;
            }
            uint32_t overflow_bit = carry;

            // Inject bit
            uint32_t p_bit = (p[i/32] >> (i%32)) & 1;
            remainder.limbs[0] |= p_bit;

            // Compare
            bool geq = (overflow_bit == 1);
            if (!geq) {
                #if defined(__NVCC__) && defined(__CUDA_ARCH__)
                    #pragma unroll
                #endif      
                for (int j = 15; j >= 0; j--) {
                    if (remainder.limbs[j] > modulus.limbs[j]) { geq = true; break; }
                    if (remainder.limbs[j] < modulus.limbs[j]) { geq = false; break; }
                    if (j==0) geq = true; // Equal
                }
            }

            if (geq) remainder.sub(modulus);
        }
        *this = remainder;
    }

    /**
     * @brief Negates the current value modulo N in-place.
     * Computes: this = -this mod N = (N - this) mod N.
     * * Correctly handles the factor -1.
     * If this == 0, the result remains 0.
     * Assumes *this < N.
     */
    __host__ __device__ void negate_mod_inplace(const uint512& N) {
        // -0 mod N is 0
        if (this->is_zero()) return;

        // -x mod N is N - x
        // Create a copy of N and subtract *this from it.
        // Since *this < N (and *this > 0), no underflow occurs.
        uint512 tmp = N;
        tmp.sub(*this);
        *this = tmp;
    }

    // =========================================================================
    // Optimized Small Type Arithmetic
    // =========================================================================

    /**
     * @brief Computes (this * x + b) % N efficiently.
     * Handles negative x and optimization for small results (common in MPQS for "(ax+b)").
     * * @param x The signed scalar multiplier.
     * @param b The additive term (must be < N).
     * @param N The modulus.
     */
    __host__ __device__ void mul_add_mod_signed(int64_t x, const uint512& b, const uint512& N) {
        bool neg = x < 0;
        uint64_t abs_x = neg ? -x : x;

        // 1. Compute prod = this * |x|
        // We capture the carry to ensure overflow safety.
        uint64_t carry = this->mul_uint64_inplace(abs_x);

        // 2. Optimization: If result fits in 512 bits (Typical MPQS case)
        if (carry == 0) {
            // fast path
            if (!neg) {
                // Case: a|x| + b
                this->add(b); // this is now a|x| + b
                // Only reduce if we exceeded N (or wrapped 512, which add handles)
                if (*this >= N) {
                    this->sub(N); // Try simple subtraction first
                    if (*this >= N) this->mod(N); // Fallback to div if still huge (rare)
                }
            } else {
                // Case: b - a|x| (mathematically)
                // We have 'this' = a|x|.
                if (b >= *this) {
                    // Result is positive: b - a|x|
                    uint512 tmp = b;
                    tmp.sub(*this);
                    *this = tmp;
                } else {
                    // Result is negative: -(a|x| - b)
                    // Modulo N: N - (a|x| - b)
                    this->sub(b); // a|x| - b
                    // this is now (a|x| - b), which is positive and < N (since carry=0)
                    this->negate_mod_inplace(N); 
                }
            }
        } else {
            // 3. Slow Path: 512-bit overflow occurred (a*x >= 2^512).
            // This is extremely rare in MPQS (requires a ~ N and x large).
            // We must perform full reduction.
            
            // Current 'this' contains (a*x) mod 2^512.
            // Real value = 'this' + carry * 2^512.
            
            // Reduce 'this' mod N
            this->mod(N); 
            
            // Calculate High Part: carry * 2^512 mod N
            // 2^512 mod N == (0 - N) in 512-bit arithmetic.
            uint512 R_mod_N((uint32_t)0);
            R_mod_N.sub(N);
            
            // We reuse mul_uint64 logic for the high part correction
            // overflow here is impossible because R_mod_N < N and carry is u64
            R_mod_N.mul_uint64_inplace(carry); 
            R_mod_N.mod(N); // Just to be safe, though carry*R_mod_N might be > N.

            this->add(R_mod_N);
            if (*this >= N) this->sub(N);

            // Now we have (a|x|) mod N in 'this'.
            if (!neg) {
                this->add_mod(b, N);
            } else {
                // b - a|x| mod N
                uint512 tmp = b;
                tmp.sub_mod(*this, N);
                *this = tmp;
            }
        }
    }    

    // Divides *this by u32, updates *this to quotient, returns remainder
    __host__ __device__ uint32_t div_uint32_inplace(uint32_t divisor) {
        if (divisor == 0) return 0;
        uint64_t rem = 0;
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 15; i >= 0; i--) {
            uint64_t cur = (rem << 32) | limbs[i];
            limbs[i] = (uint32_t)(cur / divisor);
            rem = cur % divisor;
        }
        return (uint32_t)rem;
    }

    // Divides *this by u64, updates *this to quotient, returns remainder
    // Uses unsigned __int128 for efficiency on CUDA devices
    __host__ __device__ uint64_t div_uint64_inplace(uint64_t divisor) {
        if (divisor == 0) return 0;
        unsigned __int128 rem = 0;
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 15; i >= 0; i--) {
            unsigned __int128 cur = (rem << 32) | limbs[i];
            limbs[i] = (uint32_t)(cur / divisor);
            rem = cur % divisor;
        }
        return (uint64_t)rem;
    }

    // Read-only Modulo u32
    __host__ __device__ uint32_t mod_uint32(uint32_t divisor) const {
        if (divisor == 0) return 0;
        uint64_t rem = 0;
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 15; i >= 0; i--) {
            rem = ((rem << 32) | limbs[i]) % divisor;
        }
        return (uint32_t)rem;
    }

    // Read-only Modulo u64 (non-mutating; does NOT alter *this).
    // Uses unsigned __int128 for the running remainder so that the
    // ((rem << 32) | limb) step cannot overflow when divisor > 2^32
    // (the 64-bit rem of mod_uint32 would wrap for 64-bit divisors).
    __host__ __device__ uint64_t mod_uint64(uint64_t d) const {
        if (d == 0) return 0;
        unsigned __int128 rem = 0;
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif
        for (int i = 15; i >= 0; i--) {
            rem = (((unsigned __int128)rem << 32) | limbs[i]) % d;
        }
        return (uint64_t)rem;
    }

    /**
     * @brief In-place multiply by a 64-bit integer.
     * Returns the overflow (carry out of 512 bits).
     * Much faster than full uint512 multiplication.
     */
    __host__ __device__ uint64_t mul_uint64_inplace(uint64_t v) {
        uint64_t carry = 0;
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 0; i < 16; i++) {
            // Wide multiply: 32x64 -> 96 bit result max, stored in u128
            unsigned __int128 prod = (unsigned __int128)limbs[i] * v + carry;
            limbs[i] = (uint32_t)prod;
            carry = (uint64_t)(prod >> 32);
        }
        return carry;
    }

    // =========================================================================
    // Helpers & Bit Logic
    // =========================================================================

    // additive_inverse_mod_n removed — use negate_mod_inplace (which assumes *this < N)
    // or reduce first then call negate_mod_inplace.

    __host__ __device__ bool is_zero() const {
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for(int i=0; i<16; i++) if(limbs[i] != 0) return false;
        return true;
    }

    __host__ __device__ bool is_one() const {
        if (limbs[0] != 1) return false;
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for(int i=1; i<16; i++) if(limbs[i] != 0) return false;
        return true;
    }

    /// @brief Returns index of the most significant set bit (0-511).
    /// @note Returns 0 for both zero and value 1 (bit 0 is MSB in both cases).
    ///       Callers that need to distinguish zero should check is_zero() first.
    __host__ __device__ int msb() const {
        for (int i = 15; i >= 0; i--) {
            if (limbs[i] != 0) {
                return (31 - clz32(limbs[i])) + (i * 32);
            }
        }
        return 0;
    }

    __host__ __device__ int countr_zero() const {
        for (int i = 0; i < 16; i++) {
            if (limbs[i] != 0) {
                return ctz32(limbs[i]) + (i * 32);
            }
        }
        return 512;
    }

    __host__ __device__ void rshift(int bits) {
        if (bits == 0) return;
        if (bits >= 512) { *this = uint512((uint32_t)0); return; }
        
        int limb_shift = bits / 32;
        int bit_shift = bits % 32;
        
        if (limb_shift > 0) {
            for (int i = 0; i < 16 - limb_shift; i++) limbs[i] = limbs[i+limb_shift];
            for (int i = 16 - limb_shift; i < 16; i++) limbs[i] = 0;
        }
        if (bit_shift > 0) {
            for (int i = 0; i < 15; i++) {
                limbs[i] = (limbs[i] >> bit_shift) | (limbs[i+1] << (32 - bit_shift));
            }
            limbs[15] >>= bit_shift;
        }
    }

    __host__ __device__ void lshift(int bits) {
        if (bits == 0) return;
        if (bits >= 512) { *this = uint512((uint32_t)0); return; }
        
        int limb_shift = bits / 32;
        int bit_shift = bits % 32;

        if (limb_shift > 0) {
            for (int i = 15; i >= limb_shift; i--) limbs[i] = limbs[i - limb_shift];
            for (int i = 0; i < limb_shift; i++) limbs[i] = 0;
        }
        if (bit_shift > 0) {
            for (int i = 15; i > 0; i--) {
                limbs[i] = (limbs[i] << bit_shift) | (limbs[i-1] >> (32 - bit_shift));
            }
            limbs[0] <<= bit_shift;
        }
    }

    // =========================================================================
    // SIGNED int512 helpers
    // =========================================================================

    // Used to determine the signe if we implicitly cast uint512 to int512.
    __host__ __device__ __forceinline__ bool msb_is_set() const {
        return (limbs[15] & 0x80000000u) != 0;
    }

    // Interprets *this as signed two’s complement; returns |this|
    // and writes sign = +1 or -1 (or 0 if value is zero, optional).
    __host__ __device__ __forceinline__ uint512 abs_twos_complement(int8_t &sign) const {
        if (is_zero()) { sign = 0; return *this; }
	if (!msb_is_set()) { sign = +1; return *this; }
	sign = -1;
	uint512 mag(uint32_t(0));
	mag.sub(*this); // 0 - v (mod 2^512) == two’s complement abs
	return mag;
    }

    // =========================================================================
    // Number Theoretic Functions (Internal)
    // =========================================================================

    __host__ __device__ uint512 sqrt() const {
        if (is_zero()) return uint512((uint32_t)0);
        uint512 x;
        // Guess: 2^(msb/2)
        int b = msb() / 2;
        x.limbs[b / 32] = (1U << (b % 32));

        // Newton-Raphson: x = (x + n/x) / 2
        for (int i = 0; i < 32; i++) {
            uint512 t = *this;
            t.div(x);
            t.add(x);
            t.rshift(1); // / 2
            
            if (!t.operator<(x)) return x; // Terminate if x stops decreasing
            x = t;
        }
        return x;
    }

    // =========================================================================
    // Operator Overloading (Exhaustive)
    // =========================================================================

    // --- Comparison (Base) ---
    __host__ __device__ bool operator==(const uint512& o) const {
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for(int i=0; i<16; i++) if(limbs[i] != o.limbs[i]) return false;
        return true;
    }
    __host__ __device__ bool operator!=(const uint512& o) const { return !(*this == o); }

    __host__ __device__ bool operator<(const uint512& o) const {
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for(int i=15; i>=0; i--) {
            if(limbs[i] != o.limbs[i]) return limbs[i] < o.limbs[i];
        }
        return false;
    }
    __host__ __device__ bool operator>(const uint512& o) const { return o < *this; }
    __host__ __device__ bool operator<=(const uint512& o) const { return !(*this > o); }
    __host__ __device__ bool operator>=(const uint512& o) const { return !(*this < o); }

    // --- Mixed Comparison (uint32) ---
    __host__ __device__ bool operator==(uint32_t v) const {
        if(limbs[0] != v) return false;
        for(int i=1; i<16; i++) if(limbs[i] != 0) return false;
        return true;
    }
    __host__ __device__ bool operator<(uint32_t v) const {
        for(int i=15; i>=1; i--) if(limbs[i] != 0) return false; // Too big
        return limbs[0] < v;
    }
    __host__ __device__ bool operator>(uint32_t v) const {
        for(int i=15; i>=1; i--) if(limbs[i] != 0) return true;
        return limbs[0] > v;
    }
    __host__ __device__ bool operator!=(uint32_t v) const { return !(*this == v); }
    __host__ __device__ bool operator<=(uint32_t v) const { return !(*this > v); }
    __host__ __device__ bool operator>=(uint32_t v) const { return !(*this < v); }

    // --- Mixed Comparison (uint64) ---
    __host__ __device__ bool operator==(uint64_t v) const {
        if(limbs[0] != (uint32_t)v) return false;
        if(limbs[1] != (uint32_t)(v>>32)) return false;
        for(int i=2; i<16; i++) if(limbs[i] != 0) return false;
        return true;
    }
    __host__ __device__ bool operator<(uint64_t v) const {
        for(int i=15; i>=2; i--) if(limbs[i] != 0) return false;
        uint64_t my_low = limbs[0] | ((uint64_t)limbs[1] << 32);
        return my_low < v;
    }
    __host__ __device__ bool operator>(uint64_t v) const {
        for(int i=15; i>=2; i--) if(limbs[i] != 0) return true;
        uint64_t my_low = limbs[0] | ((uint64_t)limbs[1] << 32);
        return my_low > v;
    }
    // Symmetrical friends for logic
    friend __host__ __device__ bool operator<(uint64_t a, const uint512& b) { return b > a; }
    friend __host__ __device__ bool operator>(uint64_t a, const uint512& b) { return b < a; }

    // --- Arithmetic Operators ---
    __host__ __device__ uint512 operator+(const uint512& b) const { uint512 r=*this; r.add(b); return r; }
    __host__ __device__ uint512& operator+=(const uint512& b) { this->add(b); return *this; }
    
    __host__ __device__ uint512 operator-(const uint512& b) const { uint512 r=*this; r.sub(b); return r; }
    __host__ __device__ uint512& operator-=(const uint512& b) { this->sub(b); return *this; }

    __host__ __device__ uint512 operator*(const uint512& b) const { uint512 r=*this; r.mult(b); return r; }
    __host__ __device__ uint512& operator*=(const uint512& b) { this->mult(b); return *this; }

    // Mixed Mult uint32
    __host__ __device__ uint512 operator*(uint32_t v) const { uint512 r=*this; r.mult_uint32(v); return r; }
    __host__ __device__ uint512& operator*=(uint32_t v) { this->mult_uint32(v); return *this; }
    friend __host__ __device__ uint512 operator*(uint32_t a, const uint512& b) { return b * a; }

    // Mixed Mult uint64 (Converts u64 to u512 for code reuse, compiler optimizes zeros)
    __host__ __device__ uint512 operator*(uint64_t v) const { 
        uint512 r = *this; 
        uint512 v512(v); 
        r.mult(v512); 
        return r; 
    }
    __host__ __device__ uint512& operator*=(uint64_t v) { 
        uint512 v512(v); 
        this->mult(v512); 
        return *this; 
    }
    friend __host__ __device__ uint512 operator*(uint64_t a, const uint512& b) { return b * a; }

    // Division / Mod
    __host__ __device__ uint512 operator/(const uint512& b) const { uint512 q=*this; q.div(b); return q; }
    __host__ __device__ uint512& operator/=(const uint512& b) { this->div(b); return *this; }

    __host__ __device__ uint512 operator%(const uint512& b) const { uint512 r=*this; r.mod(b); return r; }
    __host__ __device__ uint512& operator%=(const uint512& b) { this->mod(b); return *this; }
    
    // Mixed Division uint32
    __host__ __device__ uint512 operator/(uint32_t v) const { uint512 q=*this; q.div_uint32_inplace(v); return q; }
    __host__ __device__ uint512& operator/=(uint32_t v) { this->div_uint32_inplace(v); return *this; }
    __host__ __device__ uint32_t operator%(uint32_t v) const { return this->mod_uint32(v); }

    // Mixed Division uint64
    __host__ __device__ uint512 operator/(uint64_t v) const { uint512 q=*this; q.div_uint64_inplace(v); return q; }
    __host__ __device__ uint512& operator/=(uint64_t v) { this->div_uint64_inplace(v); return *this; }
    // Note: operator% for uint64 returns uint64, not uint512
    __host__ __device__ uint64_t operator%(uint64_t v) const {
        uint512 tmp = *this;
        return tmp.div_uint64_inplace(v);
    }

    // Casting to uint128
    __host__ __device__ bool fits_in_128() const {
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for(int i=4; i<16; i++) if(limbs[i] != 0) return false;
	return true;
    }

    __host__ __device__ unsigned __int128 to_uint128() const {
        unsigned __int128 lo = limbs[0] | ((unsigned __int128)limbs[1] << 32);
	unsigned __int128 hi = limbs[2] | ((unsigned __int128)limbs[3] << 32);
	return lo | (hi << 64);
    }    

    // =========================================================================
    // String conversions
    // =========================================================================

    /**
     * @brief Formats the number as a decimal string (Host only).
     * Exploits 32-bit division for efficiency (base 10^9).
     */
    __host__ std::string to_string() const {
        // Handle zero case explicitly
        if (this->is_zero()) return "0";

        uint512 temp = *this;
        // 2^512 is approx 1.34 x 10^154. 
        // We extract 9 digits at a time. 154 / 9 = 17.1 chunks.
        // A fixed buffer of 20 is safe and avoids dynamic allocation overhead.
        uint32_t chunks[20]; 
        int count = 0;
        const uint32_t divisor = 1000000000; // 10^9

        while (!temp.is_zero()) {
            // Get the remainder of division by 10^9 (next 9 digits)
            // Simultaneously divide number by 10^9 in-place
	    chunks[count++] = temp.div_uint32_inplace(divisor);
        }

        std::stringstream ss;
        // Print the most significant chunk (no leading zeros)
        ss << chunks[count - 1];

        // Print remaining chunks with 9-digit zero padding
        for (int i = count - 2; i >= 0; i--) {
            ss << std::setw(9) << std::setfill('0') << chunks[i];
        }

        return ss.str();
    }
    
    /**
     * @brief Formats the number as a hex string (Host only).
     * Format: "0x" followed by big-endian 8-digit hex limbs (most-significant first).
     */
    __host__ std::string to_hex_string() const {
        std::stringstream ss;
        ss << "0x";

        bool nonzero = false;
        for (int i = 15; i >= 0; i--) {
            if (limbs[i] != 0 || nonzero) {
                // Use std::hex and set width to 8 with '0' padding to match printf("%08x")
                ss << std::hex << std::setw(8) << std::setfill('0') << limbs[i];
                nonzero = true;
            }
        }

        if (!nonzero) {
            ss << "0";
        }

        return ss.str();
    }

};

} // namespace mpqs
