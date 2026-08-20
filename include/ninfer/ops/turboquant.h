#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::turboquant {

// Qwen3.6 uses a 256-wide attention head.  A row is recursively encoded as
// 255 PolarQuant angles at three bits each plus one FP16 radius.  Keys append a
// one-bit structured-QJL sketch (m=256 random hyperplanes) and its FP16 residual norm. The packed
// payload is 3.5 bits/value before the two row-scale values (3.59375 bits/value including them).
inline constexpr std::int32_t kHeadDim           = 256;
inline constexpr std::int32_t kAngleCount        = 255;
inline constexpr std::int32_t kAngleBits         = 3;
inline constexpr std::int32_t kAngleBytes        = 96;
inline constexpr std::int32_t kRadiusOffset      = 96;
inline constexpr std::int32_t kQjlSignsOffset    = 98;
inline constexpr std::int32_t kQjlProjectionDim  = 256;
inline constexpr std::int32_t kQjlSignBytes      = 32;
inline constexpr std::int32_t kQjlNormOffset     = 130;
inline constexpr std::int32_t kKeyBytes          = 132;
inline constexpr std::int32_t kValueBytes        = 98;

// Diagnostics for the decode attention window guard.  The TurboQuant decode kernel used to derive
// its key range straight from a cache position, so a position at or beyond the logical capacity
// drove the tile loop past the block table and the kernel never retired (Xid 109 CTX SWITCH
// TIMEOUT).  It now rejects such a position the way the INT8 and BF16 decode kernels already did;
// these counters record whether that path is ever reached.
struct DecodeWindowDiagnostics {
    std::uint64_t rejected_positions;
    std::int32_t  max_rejected_position;
    std::int32_t  logical_capacity;
};

[[nodiscard]] DecodeWindowDiagnostics decode_window_diagnostics();
void reset_decode_window_diagnostics();

[[nodiscard]] constexpr std::size_t payload_bytes(std::uint32_t tokens,
                                                  std::uint32_t kv_heads,
                                                  std::uint32_t layers) noexcept {
    return static_cast<std::size_t>(tokens) * kv_heads * layers *
           static_cast<std::size_t>(kKeyBytes + kValueBytes);
}

} // namespace ninfer::ops::turboquant
