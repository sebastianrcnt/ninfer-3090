#pragma once

// ninfer::runtime - the NINFSLOT container magic shared by persisted-conversation formats.
//
// v1..v3 were single-lane KV slots behind llama.cpp's /slots surface; that lane-oriented
// persistence path was replaced by the tiered conversation checkpoint cache (v4, see
// conversation_disk.h). The magic carries over because both formats describe the same kind of
// payload: packed KV plus GDN state bound to one exact artifact identity.

#include <cstdint>

namespace ninfer::runtime {

inline constexpr char kSlotFileMagic[8] = {'N', 'I', 'N', 'F', 'S', 'L', 'O', 'T'};

} // namespace ninfer::runtime
