#pragma once

#include "ninfer/ops/turboquant.h"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <mma.h>

#include <cstdint>

namespace ninfer::ops {

namespace tq = turboquant;

// Namespace-scope constant tables are essential here.  A function-local constexpr table with
// runtime indices is materialized on the per-thread stack by ptxas, and the decoder calls these
// lookups for every cached row.  Constant memory keeps the immutable 768-byte codebook resident.
static __device__ __constant__ float kTqCentroid[8][8] = {
        {0.392699082f, 1.178097245f, 1.963495408f, 2.748893572f, 3.534291735f,
         4.319689899f, 5.105088062f, 5.890486225f},
        {0.188449608f, 0.379996302f, 0.548254940f, 0.707229964f, 0.863566363f,
         1.022541387f, 1.190800025f, 1.382346719f},
        {0.307516578f, 0.473175544f, 0.606067952f, 0.726746549f, 0.844049778f,
         0.964728374f, 1.097620783f, 1.263279749f},
        {0.425761282f, 0.555650320f, 0.654797058f, 0.742870236f, 0.827926090f,
         0.915999269f, 1.015146007f, 1.145035044f},
        {0.523554840f, 0.620028927f, 0.691886908f, 0.755015484f, 0.815780843f,
         0.878909419f, 0.950767400f, 1.047241487f},
        {0.597674262f, 0.667508733f, 0.718909133f, 0.763819094f, 0.806977233f,
         0.851887194f, 0.903287594f, 0.973122065f},
        {0.651770691f, 0.701715286f, 0.738262509f, 0.770108630f, 0.800687697f,
         0.832533818f, 0.869081041f, 0.919025636f},
        {0.690600431f, 0.726114351f, 0.752027142f, 0.774576457f, 0.796219870f,
         0.818769185f, 0.844681976f, 0.880195896f},
};

__device__ __forceinline__ float tq_centroid(int level, int code) {
    return kTqCentroid[level - 1][code];
}

static __device__ __constant__ float kTqSinCentroid[8][8] = {
        {0.382683433f, 0.923879532f, 0.923879533f, 0.382683432f, -0.382683432f,
         -0.923879533f, -0.923879533f, -0.382683433f},
        {0.187336177f, 0.370917035f, 0.521198727f, 0.649730583f, 0.760164567f,
         0.853435344f, 0.928666007f, 0.982295860f},
        {0.302692652f, 0.455715211f, 0.569640145f, 0.664441721f, 0.747340083f,
         0.821894218f, 0.890125635f, 0.953088222f},
        {0.413014220f, 0.527495896f, 0.608998288f, 0.676404710f, 0.736530154f,
         0.793171536f, 0.849557579f, 0.910724576f},
        {0.499961950f, 0.581058703f, 0.637991318f, 0.685299945f, 0.728260932f,
         0.770043556f, 0.813861649f, 0.866047371f},
        {0.562721434f, 0.619031353f, 0.658564160f, 0.691684631f, 0.722199675f,
         0.752524583f, 0.785366274f, 0.826646592f},
        {0.606595075f, 0.645528662f, 0.673003812f, 0.696213221f, 0.717835044f,
         0.739639013f, 0.763736045f, 0.795010953f},
        {0.637000147f, 0.663969122f, 0.683120596f, 0.699413425f, 0.714717330f,
         0.730305588f, 0.747759992f, 0.770863679f},
};

__device__ __forceinline__ float tq_sin_centroid(int level, int code) {
    return kTqSinCentroid[level - 1][code];
}

static __device__ __constant__ float kTqCosCentroid[8][8] = {
        {0.923879532f, 0.382683432f, -0.382683432f, -0.923879533f, -0.923879533f,
         -0.382683432f, 0.382683432f, 0.923879532f},
        {0.982295860f, 0.928666007f, 0.853435344f, 0.760164567f, 0.649730583f,
         0.521198727f, 0.370917035f, 0.187336177f},
        {0.953088222f, 0.890125635f, 0.821894218f, 0.747340082f, 0.664441721f,
         0.569640145f, 0.455715211f, 0.302692652f},
        {0.910724577f, 0.849557579f, 0.793171536f, 0.736530154f, 0.676404711f,
         0.608998288f, 0.527495896f, 0.413014220f},
        {0.866047371f, 0.813861649f, 0.770043556f, 0.728260932f, 0.685299945f,
         0.637991318f, 0.581058703f, 0.499961950f},
        {0.826646592f, 0.785366274f, 0.752524583f, 0.722199675f, 0.691684631f,
         0.658564160f, 0.619031353f, 0.562721433f},
        {0.795010953f, 0.763736045f, 0.739639013f, 0.717835044f, 0.696213221f,
         0.673003812f, 0.645528662f, 0.606595075f},
        {0.770863679f, 0.747759992f, 0.730305588f, 0.714717330f, 0.699413424f,
         0.683120596f, 0.663969122f, 0.637000147f},
};

__device__ __forceinline__ float tq_cos_centroid(int level, int code) {
    return kTqCosCentroid[level - 1][code];
}

// The block decoder needs both children at levels 1..4. Keep each exact pair adjacent so one
// 64-bit constant load replaces separate, divergent sin and cos transactions.
static __device__ __constant__ float2 kTqChildCentroid[4][8] = {
    {{0.923879532f, 0.382683433f},
     {0.382683432f, 0.923879532f},
     {-0.382683432f, 0.923879533f},
     {-0.923879533f, 0.382683432f},
     {-0.923879533f, -0.382683432f},
     {-0.382683432f, -0.923879533f},
     {0.382683432f, -0.923879533f},
     {0.923879532f, -0.382683433f}},
    {{0.982295860f, 0.187336177f},
     {0.928666007f, 0.370917035f},
     {0.853435344f, 0.521198727f},
     {0.760164567f, 0.649730583f},
     {0.649730583f, 0.760164567f},
     {0.521198727f, 0.853435344f},
     {0.370917035f, 0.928666007f},
     {0.187336177f, 0.982295860f}},
    {{0.953088222f, 0.302692652f},
     {0.890125635f, 0.455715211f},
     {0.821894218f, 0.569640145f},
     {0.747340082f, 0.664441721f},
     {0.664441721f, 0.747340083f},
     {0.569640145f, 0.821894218f},
     {0.455715211f, 0.890125635f},
     {0.302692652f, 0.953088222f}},
    {{0.910724577f, 0.413014220f},
     {0.849557579f, 0.527495896f},
     {0.793171536f, 0.608998288f},
     {0.736530154f, 0.676404710f},
     {0.676404711f, 0.736530154f},
     {0.608998288f, 0.793171536f},
     {0.527495896f, 0.849557579f},
     {0.413014220f, 0.910724576f}},
};

__device__ __forceinline__ int tq_angle_offset(int level) {
    return level == 1 ? 0 : 256 - (256 >> (level - 1));
}

__device__ __forceinline__ unsigned tq_hash(unsigned x, unsigned seed) {
    x ^= seed + 0x9e3779b9U + (x << 6U) + (x >> 2U);
    x ^= x >> 16U;
    x *= 0x7feb352dU;
    x ^= x >> 15U;
    x *= 0x846ca68bU;
    return x ^ (x >> 16U);
}

__device__ __forceinline__ float tq_random_sign(int d, unsigned seed) {
    return (tq_hash(static_cast<unsigned>(d), seed) & 1U) ? -1.0f : 1.0f;
}

// In-place H*D transform.  Normalize for the orthogonal PolarQuant rotation;
// leave unnormalized for the structured random-hyperplane QJL projection.
__device__ __forceinline__ float tq_signed_hadamard(float x, float* shared, int d, unsigned seed,
                                                    bool normalize) {
    x *= tq_random_sign(d, seed);
    shared[d] = x;
    __syncthreads();
    for (int stride = 1; stride < tq::kHeadDim; stride <<= 1) {
        const float self    = shared[d];
        const float partner = shared[d ^ stride];
        __syncthreads();
        x         = (d & stride) ? partner - self : self + partner;
        shared[d] = x;
        __syncthreads();
    }
    return normalize ? x * 0.0625f : x;
}

__device__ __forceinline__ int tq_nearest_angle(float theta, int level) {
    int best       = 0;
    float best_err = CUDART_INF_F;
#pragma unroll
    for (int code = 0; code < 8; ++code) {
        float err = fabsf(theta - tq_centroid(level, code));
        if (level == 1) { err = fminf(err, 2.0f * CUDART_PI_F - err); }
        if (err < best_err) {
            best_err = err;
            best     = code;
        }
    }
    return best;
}

__device__ __forceinline__ int tq_quantize_signed_pair(float x, float y) {
    int best = 0;
    float best_dot = -CUDART_INF_F;
#pragma unroll
    for (int code = 0; code < 8; ++code) {
        const float dot = x * tq_cos_centroid(1, code) + y * tq_sin_centroid(1, code);
        if (dot > best_dot) {
            best_dot = dot;
            best = code;
        }
    }
    return best;
}

__device__ __forceinline__ int tq_quantize_energy(float left, float right, int level) {
    constexpr float threshold[7][7] = {
        {0.085341136f, 0.250598015f, 0.526586392f, 1.000000000f, 1.899023628f,
         3.990454596f, 11.717678580f},
        {0.169298925f, 0.358699867f, 0.618461054f, 1.000000000f, 1.616916686f,
         2.787846027f, 5.906712043f},
        {0.285471245f, 0.478632877f, 0.706098232f, 0.999999998f, 1.416233539f,
         2.089283978f, 3.502979779f},
        {0.414091637f, 0.592358578f, 0.780028965f, 1.000000000f, 1.282003676f,
         1.688166659f, 2.414924409f},
        {0.537423017f, 0.690137658f, 0.838312508f, 1.000000000f, 1.192872576f,
         1.448986284f, 1.860731619f},
        {0.645272619f, 0.769213645f, 0.882557318f, 1.000000000f, 1.133070885f,
         1.300028942f, 1.549732580f},
        {0.733894691f, 0.830626218f, 0.915381611f, 1.000000000f, 1.092440561f,
         1.203910952f, 1.362593316f},
    };
    int code = 0;
#pragma unroll
    for (int boundary = 0; boundary < 7; ++boundary) {
        code += right > left * threshold[level - 2][boundary];
    }
    return code;
}

__device__ __forceinline__ int tq_get_code(const std::uint8_t* row, int angle) {
    const int bit  = angle * tq::kAngleBits;
    const int byte = bit >> 3;
    unsigned word  = row[byte];
    if (byte + 1 < tq::kAngleBytes) { word |= static_cast<unsigned>(row[byte + 1]) << 8U; }
    return static_cast<int>((word >> (bit & 7)) & 7U);
}

__device__ __forceinline__ float tq_decode_coordinate(const std::uint8_t* row, int d) {
    float radius = __half2float(*reinterpret_cast<const __half*>(row + tq::kRadiusOffset));
#pragma unroll
    for (int level = 8; level >= 2; --level) {
        const int node = d >> level;
        const int code = tq_get_code(row, tq_angle_offset(level) + node);
        radius *= ((d >> (level - 1)) & 1) ? tq_sin_centroid(level, code)
                                           : tq_cos_centroid(level, code);
    }
    const int leaf = tq_get_code(row, d >> 1);
    return radius * ((d & 1) ? tq_sin_centroid(1, leaf) : tq_cos_centroid(1, leaf));
}

__device__ __forceinline__ void tq_decode_children(const std::uint8_t* row, int level, int node,
                                                    float parent, float& left, float& right) {
    const int code = tq_get_code(row, tq_angle_offset(level) + node);
    const float2 children = kTqChildCentroid[level - 1][code];
    left                  = parent * children.x;
    right                 = parent * children.y;
}

// Decode one aligned 16-coordinate Polar subtree while sharing its root-to-level-5 prefix.
// The old float[16] result was dynamically indexed while building the tree.  ptxas consequently
// placed it on the per-thread stack, turning every cached row into hundreds of bytes of local
// memory traffic.  Keep every node as a named scalar and store the leaves directly into the MMA
// tile.  This preserves the exact level-8..1 multiplication order without a local array.
template <bool Swizzled>
__device__ __forceinline__ void tq_decode_block16_store(const std::uint8_t* row, int d_base,
                                                        __nv_bfloat16* dst, int swizzle_row = 0) {
    float prefix = __half2float(*reinterpret_cast<const __half*>(row + tq::kRadiusOffset));
#pragma unroll
    for (int level = 8; level >= 5; --level) {
        const int code = tq_get_code(row, tq_angle_offset(level) + (d_base >> level));
        prefix *= ((d_base >> (level - 1)) & 1) ? tq_sin_centroid(level, code)
                                                : tq_cos_centroid(level, code);
    }

    float l4_0, l4_1;
    tq_decode_children(row, 4, d_base >> 4, prefix, l4_0, l4_1);
    float l3_0, l3_1, l3_2, l3_3;
    tq_decode_children(row, 3, d_base >> 3, l4_0, l3_0, l3_1);
    tq_decode_children(row, 3, (d_base >> 3) + 1, l4_1, l3_2, l3_3);
    float l2_0, l2_1, l2_2, l2_3, l2_4, l2_5, l2_6, l2_7;
    tq_decode_children(row, 2, d_base >> 2, l3_0, l2_0, l2_1);
    tq_decode_children(row, 2, (d_base >> 2) + 1, l3_1, l2_2, l2_3);
    tq_decode_children(row, 2, (d_base >> 2) + 2, l3_2, l2_4, l2_5);
    tq_decode_children(row, 2, (d_base >> 2) + 3, l3_3, l2_6, l2_7);
    float v0, v1, v2, v3, v4, v5, v6, v7;
    float v8, v9, v10, v11, v12, v13, v14, v15;
    tq_decode_children(row, 1, d_base >> 1, l2_0, v0, v1);
    tq_decode_children(row, 1, (d_base >> 1) + 1, l2_1, v2, v3);
    tq_decode_children(row, 1, (d_base >> 1) + 2, l2_2, v4, v5);
    tq_decode_children(row, 1, (d_base >> 1) + 3, l2_3, v6, v7);
    tq_decode_children(row, 1, (d_base >> 1) + 4, l2_4, v8, v9);
    tq_decode_children(row, 1, (d_base >> 1) + 5, l2_5, v10, v11);
    tq_decode_children(row, 1, (d_base >> 1) + 6, l2_6, v12, v13);
    tq_decode_children(row, 1, (d_base >> 1) + 7, l2_7, v14, v15);

    const int offset0 = Swizzled ? gqa_prefill_swz(swizzle_row, d_base) : d_base;
    const int offset1 = Swizzled ? gqa_prefill_swz(swizzle_row, d_base + 8) : d_base + 8;
    const int4 lo = make_int4(static_cast<int>(pack_bf16x2(v0, v1)),
                              static_cast<int>(pack_bf16x2(v2, v3)),
                              static_cast<int>(pack_bf16x2(v4, v5)),
                              static_cast<int>(pack_bf16x2(v6, v7)));
    const int4 hi = make_int4(static_cast<int>(pack_bf16x2(v8, v9)),
                              static_cast<int>(pack_bf16x2(v10, v11)),
                              static_cast<int>(pack_bf16x2(v12, v13)),
                              static_cast<int>(pack_bf16x2(v14, v15)));
    store_vec(dst + offset0, lo);
    store_vec(dst + offset1, hi);
}

template <bool Key>
__device__ __forceinline__ void tq_encode_row(const __nv_bfloat16* src, std::uint8_t* row,
                                              float* values, float* energy, std::uint8_t* codes,
                                              int d) {
    constexpr unsigned kPolarSeed = 0x504f4c52U;
    constexpr unsigned kQjlSeed   = 0x514a4c31U;
    float x                       = __bfloat162float(src[d]);
    x                             = tq_signed_hadamard(x, values, d, kPolarSeed, true);
    values[d]                     = x;
    energy[d]                     = x * x;
    __syncthreads();

    int input = 0;
    for (int level = 1; level <= 8; ++level) {
        const int nodes = tq::kHeadDim >> level;
        const int out   = input ^ tq::kHeadDim;
        if (d < nodes) {
            const float left  = energy[input + 2 * d];
            const float right = energy[input + 2 * d + 1];
            const int code = level == 1 ? tq_quantize_signed_pair(values[2 * d], values[2 * d + 1])
                                        : tq_quantize_energy(left, right, level);
            codes[tq_angle_offset(level) + d] = static_cast<std::uint8_t>(code);
            energy[out + d] = left + right;
        }
        __syncthreads();
        input = out;
    }

    const __half radius_bits = __float2half_rn(sqrtf(fmaxf(energy[input], 0.0f)));
    if (d < tq::kAngleBytes) {
        const int first_bit = d * 8;
        const int a0        = first_bit / tq::kAngleBits;
        unsigned packed     = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int angle = a0 + j;
            if (angle >= tq::kAngleCount) { break; }
            const int bit = angle * tq::kAngleBits - first_bit;
            if (bit >= 8) { break; }
            const unsigned code = codes[angle];
            if (bit >= 0) {
                packed |= code << bit;
            } else {
                packed |= code >> (-bit);
            }
        }
        row[d] = static_cast<std::uint8_t>(packed & 0xffU);
    }
    if (d == 0) { *reinterpret_cast<__half*>(row + tq::kRadiusOffset) = radius_bits; }
    __syncthreads();

    if constexpr (Key) {
        const float reconstructed = tq_decode_coordinate(row, d);
        const float residual      = values[d] - reconstructed;
        energy[d]                 = residual * residual;
        values[d]                 = residual;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (d < stride) { energy[d] += energy[d + stride]; }
            __syncthreads();
        }
        const __half norm_bits = __float2half_rn(sqrtf(fmaxf(energy[0], 0.0f)));
        const float projection = tq_signed_hadamard(residual, values, d, kQjlSeed, false);
        values[d]              = projection;
        __syncthreads();
        if (d < tq::kQjlSignBytes) {
            unsigned bits = 0;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                if (values[8 * d + b] >= 0.0f) { bits |= 1U << b; }
            }
            row[tq::kQjlSignsOffset + d] = static_cast<std::uint8_t>(bits);
        }
        if (d == 0) { *reinterpret_cast<__half*>(row + tq::kQjlNormOffset) = norm_bits; }
        __syncthreads();
    }
}

template <typename Geometry>
__launch_bounds__(256) __global__ void gqa_turboquant_append_batch_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v, const std::int32_t* positions,
    const std::int32_t* valid_columns, const std::int32_t* table_rows,
    const std::int32_t* block_tables, int table_stride, std::uint8_t* cache_k,
    std::uint8_t* cache_v, int full_width, int column_begin, int width, int batch_size) {
    const int batch = static_cast<int>(blockIdx.z);
    const int unit  = static_cast<int>(blockIdx.x);
    const int token = unit / Geometry::KVHeads;
    const int kvh   = unit - token * Geometry::KVHeads;
    if (batch >= batch_size || token >= width ||
        (valid_columns != nullptr && column_begin + token >= valid_columns[batch])) {
        return;
    }
    const int column = column_begin + token + batch * full_width;
    const int d      = static_cast<int>(threadIdx.x);
    const int position = positions[column];
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* table = block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    const int physical = table[position >> kPagedKVPageShift];
    const int page_off = position & kPagedKVPageMask;
    std::uint8_t* krow = cache_k + paged_kv_element_offset<tq::kKeyBytes, Geometry::KVHeads>(
                                         physical, kvh, page_off, 0);
    std::uint8_t* vrow = cache_v + paged_kv_element_offset<tq::kValueBytes, Geometry::KVHeads>(
                                         physical, kvh, page_off, 0);
    const std::int64_t src_base = static_cast<std::int64_t>(tq::kHeadDim) *
                                  (kvh + Geometry::KVHeads * column);
    __shared__ float values[tq::kHeadDim];
    __shared__ float energy[2 * tq::kHeadDim];
    __shared__ std::uint8_t codes[tq::kHeadDim];
    tq_encode_row<true>(k + src_base, krow, values, energy, codes, d);
    tq_encode_row<false>(v + src_base, vrow, values, energy, codes, d);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__ void gqa_turboquant_append_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v, const std::int32_t* positions,
    Metadata metadata, std::uint8_t* cache_k, std::uint8_t* cache_v, int width) {
    const int valid = metadata.valid_tokens(width);
    const int unit  = static_cast<int>(blockIdx.x);
    const int token = unit / Geometry::KVHeads;
    const int kvh   = unit - token * Geometry::KVHeads;
    if (token >= valid) { return; }
    const int d        = static_cast<int>(threadIdx.x);
    const int position = positions[token];
    const int physical = metadata.block_table()[position >> kPagedKVPageShift];
    const int page_off = position & kPagedKVPageMask;
    std::uint8_t* krow = cache_k + paged_kv_element_offset<tq::kKeyBytes, Geometry::KVHeads>(
                                         physical, kvh, page_off, 0);
    std::uint8_t* vrow = cache_v + paged_kv_element_offset<tq::kValueBytes, Geometry::KVHeads>(
                                         physical, kvh, page_off, 0);
    const std::int64_t src_base =
        static_cast<std::int64_t>(tq::kHeadDim) * (kvh + Geometry::KVHeads * token);
    __shared__ float values[tq::kHeadDim];
    __shared__ float energy[2 * tq::kHeadDim];
    __shared__ std::uint8_t codes[tq::kHeadDim];
    tq_encode_row<true>(k + src_base, krow, values, energy, codes, d);
    tq_encode_row<false>(v + src_base, vrow, values, energy, codes, d);
}

template <typename Geometry>
__launch_bounds__(256, 2) __global__ void gqa_turboquant_attention_kernel(
    const __nv_bfloat16* q, const std::int32_t* positions, const std::uint8_t* cache_k,
    const std::uint8_t* cache_v, const std::int32_t* block_tables,
    const std::int32_t* valid_columns, const std::int32_t* table_rows, int table_stride,
    int full_width, int column_begin, int tokens, int batch_size, int logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    constexpr unsigned kPolarSeed = 0x504f4c52U;
    constexpr unsigned kQjlSeed   = 0x514a4c31U;
    constexpr float kQjlFactor    = 0.00489575835f; // sqrt(pi/2) / m, m=256
    const int kvh      = static_cast<int>(blockIdx.x);
    const int split    = static_cast<int>(blockIdx.y);
    const int flat     = static_cast<int>(blockIdx.z);
    const int batch    = flat / tokens;
    const int token    = flat - batch * tokens;
    const int splits   = static_cast<int>(gridDim.y);
    const int d        = static_cast<int>(threadIdx.x);
    if (batch_size > 1) {
        partial_acc += static_cast<std::int64_t>(batch) * tq::kHeadDim * Geometry::QHeads *
                       tokens * splits;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * splits;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * splits;
    }
    int valid_tokens = tokens;
    if (valid_columns != nullptr) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : min(remaining, tokens);
    }
    auto write_neutral = [&]() {
        for (int local = 0; local < Geometry::GroupSize; ++local) {
            const int qh = kvh * Geometry::GroupSize + local;
            partial_acc[gqa_partial_acc_index<Geometry>(qh, d, token, split, tokens)] =
                __float2bfloat16(0.0f);
            if (d == 0) {
                partial_m[gqa_partial_stat_index<Geometry>(qh, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(qh, token, split, tokens)] = 0.0f;
            }
        }
    };
    if (token >= valid_tokens || kvh >= Geometry::KVHeads) {
        if (kvh < Geometry::KVHeads) { write_neutral(); }
        return;
    }
    const std::int64_t column = column_begin + token +
                                static_cast<std::int64_t>(batch) * full_width;
    q += static_cast<std::int64_t>(tq::kHeadDim) * Geometry::QHeads * column;
    positions += column;
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* table = block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    const int qpos   = positions[0];
    const int window = positions[valid_tokens - 1 - token] + 1;
    if (qpos < 0 || qpos >= logical_capacity) {
        write_neutral();
        return;
    }
    const int active = gqa_small_t_active_splits<Geometry, true>(window, splits, tokens);
    if (split >= active) { return; }
    const int per_split = div_up(window, active);
    const int begin     = split * per_split;
    const int end       = min(window, begin + per_split);
    if (begin >= end) {
        write_neutral();
        return;
    }

    __shared__ float work[tq::kHeadDim];
    __shared__ float qrot[Geometry::GroupSize * tq::kHeadDim];
    __shared__ float qjl[Geometry::GroupSize * tq::kHeadDim];
    __shared__ float kvec[tq::kHeadDim];
    __shared__ float vvec[tq::kHeadDim];
    __shared__ std::uint8_t krow[tq::kKeyBytes];
    __shared__ std::uint8_t vrow[tq::kValueBytes];
    for (int local = 0; local < Geometry::GroupSize; ++local) {
        const int qh = kvh * Geometry::GroupSize + local;
        float qr = tq_signed_hadamard(__bfloat162float(q[gqa_q_index<Geometry>(qh, d)]), work, d,
                                      kPolarSeed, true);
        qrot[local * tq::kHeadDim + d] = qr;
        __syncthreads();
        qjl[local * tq::kHeadDim + d] = tq_signed_hadamard(qr, work, d, kQjlSeed, false);
        __syncthreads();
    }

    const int warp = d >> 5;
    const int lane = d & 31;
    const bool consumer = warp < Geometry::GroupSize;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { acc[i] = 0.0f; }
    float m = -CUDART_INF_F;
    float l = 0.0f;
    for (int key = begin; key < end; ++key) {
        if (key > qpos) { break; }
        const int physical = table[key >> kPagedKVPageShift];
        const int page_off = key & kPagedKVPageMask;
        const std::uint8_t* kg =
            cache_k + paged_kv_element_offset<tq::kKeyBytes, Geometry::KVHeads>(
                          physical, kvh, page_off, 0);
        const std::uint8_t* vg =
            cache_v + paged_kv_element_offset<tq::kValueBytes, Geometry::KVHeads>(
                          physical, kvh, page_off, 0);
        if (d < tq::kKeyBytes) { krow[d] = kg[d]; }
        if (d < tq::kValueBytes) { vrow[d] = vg[d]; }
        __syncthreads();
        kvec[d] = tq_decode_coordinate(krow, d);
        vvec[d] = tq_decode_coordinate(vrow, d);
        __syncthreads();
        if (consumer) {
            float dot = 0.0f;
            float qjl_dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int element = lane + 32 * i;
                dot += qrot[warp * tq::kHeadDim + element] * kvec[element];
                if (element < tq::kQjlProjectionDim) {
                    const unsigned bit =
                        (krow[tq::kQjlSignsOffset + (element >> 3)] >> (element & 7)) & 1U;
                    qjl_dot += qjl[warp * tq::kHeadDim + element] * (bit ? 1.0f : -1.0f);
                }
            }
            dot = warp_sum(dot);
            qjl_dot = warp_sum(qjl_dot);
            float alpha = 0.0f;
            float probability = 0.0f;
            if (lane == 0) {
                const float gamma =
                    __half2float(*reinterpret_cast<const __half*>(krow + tq::kQjlNormOffset));
                const float score = scale * (dot + gamma * kQjlFactor * qjl_dot);
                const float nm = fmaxf(m, score);
                alpha = m == -CUDART_INF_F ? 0.0f : expf(m - nm);
                probability = expf(score - nm);
                l = l * alpha + probability;
                m = nm;
            }
            alpha = __shfl_sync(0xffffffffU, alpha, 0);
            probability = __shfl_sync(0xffffffffU, probability, 0);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[i] = acc[i] * alpha + probability * vvec[lane + 32 * i];
            }
        }
        __syncthreads();
    }
    if (consumer) {
        const int qh = kvh * Geometry::GroupSize + warp;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int element = lane + 32 * i;
            partial_acc[gqa_partial_acc_index<Geometry>(qh, element, token, split, tokens)] =
                __float2bfloat16(acc[i]);
        }
        if (lane == 0) {
            partial_m[gqa_partial_stat_index<Geometry>(qh, token, split, tokens)] = m;
            partial_l[gqa_partial_stat_index<Geometry>(qh, token, split, tokens)] = l;
        }
    }
}

// H256 applied by one warp to eight coordinates per lane.  This is the same signed orthogonal
// transform as tq_signed_hadamard, without block-wide barriers between butterfly stages.
__device__ __forceinline__ void tq_warp_polar_query(float (&x)[8], int lane) {
    constexpr unsigned kPolarSeed = 0x504f4c52U;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int d = lane + 32 * i;
        x[i] *= tq_random_sign(d, kPolarSeed);
    }
#pragma unroll
    for (int stride = 1; stride < 32; stride <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float self = x[i];
            const float partner = __shfl_xor_sync(0xffffffffU, self, stride);
            x[i] = (lane & stride) ? partner - self : self + partner;
        }
    }
#pragma unroll
    for (int mask = 1; mask < 8; mask <<= 1) {
        float old[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) { old[i] = x[i]; }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[i] = (i & mask) ? old[i ^ mask] - old[i] : old[i] + old[i ^ mask];
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) { x[i] *= 0.0625f; }
}

// Structured QJL dual transform. If r_hat = D H sign(H D r) * c / m, then
// q dot r_hat = (H D q) dot sign(H D r) * c / m. Applying this once per query row lets the
// packed sign plane feed a second MMA directly, instead of reconstructing every cached K row.
__device__ __forceinline__ void tq_warp_qjl_query(float (&x)[8], int lane) {
    constexpr unsigned kQjlSeed = 0x514a4c31U;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int d = lane + 32 * i;
        x[i] *= tq_random_sign(d, kQjlSeed);
    }
#pragma unroll
    for (int stride = 1; stride < 32; stride <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float self = x[i];
            const float partner = __shfl_xor_sync(0xffffffffU, self, stride);
            x[i] = (lane & stride) ? partner - self : self + partner;
        }
    }
#pragma unroll
    for (int mask = 1; mask < 8; mask <<= 1) {
        float old[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) { old[i] = x[i]; }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[i] = (i & mask) ? old[i ^ mask] - old[i] : old[i] + old[i ^ mask];
        }
    }
}

// Reconstruct the QJL residual once per cached key instead of applying a second QK matrix
// multiplication for every query row.  Encoding stores s = sign(H D r), and the estimator is
// r_hat = D H s * ||r|| * c/m.  A warp owns all 256 coordinates of one row (eight per lane), so
// the Hadamard stays in registers and the reconstructed residual is folded into the transient
// Polar key tile.  This is algebraically the same estimator used by tq_warp_qjl_query, but moves
// work from O(queries * keys * D) to O(keys * D log D).
template <bool Swizzled>
__device__ __forceinline__ void tq_warp_add_qjl_residual(const std::uint8_t* row,
                                                         __nv_bfloat16* dst, int swizzle_row,
                                                         int lane) {
    constexpr unsigned kQjlSeed = 0x514a4c31U;
    constexpr float kQjlFactor  = 0.00489575835f;
    float x[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int d = lane + 32 * i;
        const unsigned bit =
            (row[tq::kQjlSignsOffset + (d >> 3)] >> (d & 7)) & 1U;
        x[i] = bit ? 1.0f : -1.0f;
    }
#pragma unroll
    for (int stride = 1; stride < 32; stride <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float self = x[i];
            const float partner = __shfl_xor_sync(0xffffffffU, self, stride);
            x[i] = (lane & stride) ? partner - self : self + partner;
        }
    }
#pragma unroll
    for (int mask = 1; mask < 8; mask <<= 1) {
        float old[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) { old[i] = x[i]; }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            x[i] = (i & mask) ? old[i ^ mask] - old[i] : old[i] + old[i ^ mask];
        }
    }
    const float gamma =
        __half2float(*reinterpret_cast<const __half*>(row + tq::kQjlNormOffset));
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int d = lane + 32 * i;
        const int index = Swizzled ? gqa_prefill_swz(swizzle_row, d) : d;
        const float residual =
            x[i] * tq_random_sign(d, kQjlSeed) * gamma * kQjlFactor;
        dst[index] = __float2bfloat16(__bfloat162float(dst[index]) + residual);
    }
}

// Small-T fused packed consumer. One CTA owns all query positions for one KV head and split, so
// every packed K/V row is decoded once for T=1..8. The reconstructed Polar+QJL key uses one SM80+
// BF16 MMA; V aggregation stays in FP32 registers so online-softmax scaling is exact per row.
template <typename Geometry, int TokenTile>
// Two resident CTAs hide constant-cache and shared-memory latency.  The centroid codebooks live
// in device constant memory, so the 128-register sm_86 budget no longer spills them to local DRAM.
__launch_bounds__(256, 2) __global__ void gqa_turboquant_mma_attention_kernel(
    const __nv_bfloat16* q, const std::int32_t* positions, const std::uint8_t* cache_k,
    const std::uint8_t* cache_v, const std::int32_t* block_tables,
    const std::int32_t* valid_columns, const std::int32_t* table_rows, int table_stride,
    int full_width, int column_begin, int batch_size, int logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    static_assert(TokenTile >= 1 && TokenTile <= 8);
    constexpr int kRows       = TokenTile * Geometry::GroupSize;
    constexpr int kMmaRows    = ((kRows + 15) / 16) * 16;
    constexpr int kPanels     = kMmaRows / 16;
    constexpr int kKeyTile    = 16;

    const int kvh    = static_cast<int>(blockIdx.x);
    const int split  = static_cast<int>(blockIdx.y);
    const int batch  = static_cast<int>(blockIdx.z);
    const int splits = static_cast<int>(gridDim.y);
    const int d      = static_cast<int>(threadIdx.x);
    const int warp   = d >> 5;
    const int lane   = d & 31;
    if (batch >= batch_size || kvh >= Geometry::KVHeads) { return; }

    if (batch_size > 1) {
        partial_acc += static_cast<std::int64_t>(batch) * tq::kHeadDim * Geometry::QHeads *
                       TokenTile * splits;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * splits;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * splits;
    }
    const int remaining = valid_columns == nullptr ? TokenTile
                                                    : valid_columns[batch] - column_begin;
    const int valid_tokens = max(0, min(TokenTile, remaining));
    const std::int64_t batch_column = static_cast<std::int64_t>(batch) * full_width + column_begin;
    const std::int32_t* batch_positions = positions + batch_column;
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* table = block_tables + static_cast<std::int64_t>(table_row) * table_stride;

    __shared__ __align__(32) __nv_bfloat16 qmatrix[kMmaRows * tq::kHeadDim];
    __shared__ __align__(32) __nv_bfloat16 tile[kKeyTile * tq::kHeadDim];
    __shared__ __align__(32) float scores[kMmaRows * kKeyTile];
    __shared__ int query_positions[TokenTile];
    __shared__ int key_begin;
    __shared__ int key_end;
    __shared__ bool live_split;

    for (int index = d; index < kMmaRows * tq::kHeadDim; index += blockDim.x) {
        qmatrix[index] = __float2bfloat16(0.0f);
    }
    if (d < TokenTile) {
        query_positions[d] = d < valid_tokens ? batch_positions[d] : -1;
    }
    if (d == 0) {
        const int window = valid_tokens == 0 ? 0 : batch_positions[valid_tokens - 1] + 1;
        const int active = window == 0
                               ? 0
                               : gqa_small_t_active_splits<Geometry, true>(window, splits,
                                                                           TokenTile);
        live_split = split < active;
        const int per_split = active == 0 ? 0 : div_up(window, active);
        key_begin = split * per_split;
        key_end   = min(window, key_begin + per_split);
    }
    __syncthreads();
    if (!live_split || key_begin >= key_end) { return; }

    // Six warps transform the six query heads concurrently; each warp retains eight dimensions
    // per lane, so no temporary BF16 expansion exists outside this CTA.
    if (warp < Geometry::GroupSize) {
        const int qh = kvh * Geometry::GroupSize + warp;
#pragma unroll
        for (int token = 0; token < TokenTile; ++token) {
            float qr[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int element = lane + 32 * i;
                const std::int64_t column = batch_column + token;
                qr[i] = token < valid_tokens && query_positions[token] >= 0 &&
                                query_positions[token] < logical_capacity
                            ? __bfloat162float(
                                  q[gqa_q_index<Geometry>(qh, element, static_cast<int>(column))])
                            : 0.0f;
            }
            tq_warp_polar_query(qr, lane);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int element = lane + 32 * i;
                qmatrix[(token * Geometry::GroupSize + warp) * tq::kHeadDim + element] =
                    __float2bfloat16(qr[i]);
            }
        }
    }
    __syncthreads();

    const bool consumer = warp < Geometry::GroupSize;
    float acc[TokenTile][8];
    float maximum[TokenTile];
    float denominator[TokenTile];
#pragma unroll
    for (int token = 0; token < TokenTile; ++token) {
        maximum[token] = -CUDART_INF_F;
        denominator[token] = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[token][i] = 0.0f; }
    }

    using namespace nvcuda;
    for (int tile_begin = key_begin; live_split && tile_begin < key_end;
         tile_begin += kKeyTile) {
        // Decode a 16-key Polar tile directly from the packed cache into transient shared MMA
        // operands. A tile never survives this loop iteration.
        {
            const int j = d >> 4;
            const int d_base = (d & 15) * 16;
            const int key = tile_begin + j;
            if (key < key_end) {
                const int physical = table[key >> kPagedKVPageShift];
                const int page_off = key & kPagedKVPageMask;
                const std::uint8_t* row =
                    cache_k + paged_kv_element_offset<tq::kKeyBytes, Geometry::KVHeads>(
                                  physical, kvh, page_off, 0);
                tq_decode_block16_store<false>(row, d_base, tile + j * tq::kHeadDim);
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    tile[j * tq::kHeadDim + d_base + i] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        // Each warp owns the two rows that its 32 threads decoded above.  Reconstruct their QJL
        // residuals in registers and fold them into the Polar tile before the sole QK MMA.
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const int j = 2 * warp + pair;
            const int key = tile_begin + j;
            if (key < key_end) {
                const int physical = table[key >> kPagedKVPageShift];
                const int page_off = key & kPagedKVPageMask;
                const std::uint8_t* row =
                    cache_k + paged_kv_element_offset<tq::kKeyBytes, Geometry::KVHeads>(
                                  physical, kvh, page_off, 0);
                tq_warp_add_qjl_residual<false>(row, tile + j * tq::kHeadDim, 0, lane);
            }
        }
        __syncthreads();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_acc;
        if (warp < kPanels) {
            wmma::fill_fragment(score_acc, 0.0f);
#pragma unroll
            for (int kk = 0; kk < tq::kHeadDim; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
                wmma::load_matrix_sync(a, qmatrix + warp * 16 * tq::kHeadDim + kk,
                                       tq::kHeadDim);
                wmma::load_matrix_sync(b, tile + kk, tq::kHeadDim);
                wmma::mma_sync(score_acc, a, b, score_acc);
            }
            wmma::store_matrix_sync(scores + warp * 16 * kKeyTile, score_acc, kKeyTile,
                                    wmma::mem_row_major);
        }
        __syncthreads();

        // Reuse the same operand tile for V; no materialized K/V buffer exists between tiles.
        {
            const int j = d >> 4;
            const int d_base = (d & 15) * 16;
            const int key = tile_begin + j;
            if (key < key_end) {
                const int physical = table[key >> kPagedKVPageShift];
                const int page_off = key & kPagedKVPageMask;
                const std::uint8_t* row =
                    cache_v + paged_kv_element_offset<tq::kValueBytes, Geometry::KVHeads>(
                                  physical, kvh, page_off, 0);
                tq_decode_block16_store<false>(row, d_base, tile + j * tq::kHeadDim);
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    tile[j * tq::kHeadDim + d_base + i] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        if (consumer) {
#pragma unroll
            for (int token = 0; token < TokenTile; ++token) {
                if (token >= valid_tokens || query_positions[token] < 0 ||
                    query_positions[token] >= logical_capacity) {
                    continue;
                }
                const int row = token * Geometry::GroupSize + warp;
#pragma unroll
                for (int j = 0; j < kKeyTile; ++j) {
                    const int key = tile_begin + j;
                    if (key >= key_end || key > query_positions[token]) { continue; }
                    const float score = scores[row * kKeyTile + j] * scale;
                    const float nm = fmaxf(maximum[token], score);
                    const float alpha = maximum[token] == -CUDART_INF_F
                                            ? 0.0f
                                            : expf(maximum[token] - nm);
                    const float probability = expf(score - nm);
                    denominator[token] = denominator[token] * alpha + probability;
                    maximum[token] = nm;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        acc[token][i] = acc[token][i] * alpha +
                                        probability * __bfloat162float(
                                                          tile[j * tq::kHeadDim + lane + 32 * i]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (consumer) {
        const int qh = kvh * Geometry::GroupSize + warp;
#pragma unroll
        for (int token = 0; token < TokenTile; ++token) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int element = lane + 32 * i;
                partial_acc[gqa_partial_acc_index<Geometry>(qh, element, token, split, TokenTile)] =
                    __float2bfloat16(acc[token][i]);
            }
            if (lane == 0) {
                partial_m[gqa_partial_stat_index<Geometry>(qh, token, split, TokenTile)] =
                    maximum[token];
                partial_l[gqa_partial_stat_index<Geometry>(qh, token, split, TokenTile)] =
                    denominator[token];
            }
        }
    }
}

template <typename Geometry>
__launch_bounds__(256) __global__ void gqa_turboquant_inverse_output_kernel(
    __nv_bfloat16* out, int width, int full_width, int column_begin, int batch_size) {
    constexpr unsigned kPolarSeed = 0x504f4c52U;
    const int unit = static_cast<int>(blockIdx.x);
    const int qh   = unit % Geometry::QHeads;
    const int tmp  = unit / Geometry::QHeads;
    const int tok  = tmp % width;
    const int b    = tmp / width;
    if (b >= batch_size) { return; }
    const int d      = static_cast<int>(threadIdx.x);
    const int column = column_begin + tok + b * full_width;
    __shared__ float values[tq::kHeadDim];
    float x = __bfloat162float(out[gqa_q_index<Geometry>(qh, d, column)]);
    values[d] = x;
    __syncthreads();
    for (int stride = 1; stride < tq::kHeadDim; stride <<= 1) {
        const float self    = values[d];
        const float partner = values[d ^ stride];
        __syncthreads();
        x         = (d & stride) ? partner - self : self + partner;
        values[d] = x;
        __syncthreads();
    }
    x = x * 0.0625f * tq_random_sign(d, kPolarSeed);
    out[gqa_q_index<Geometry>(qh, d, column)] = __float2bfloat16(x);
}

} // namespace ninfer::ops
