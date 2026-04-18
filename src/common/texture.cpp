// Copyright 2019 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include "common/assert.h"
#include "common/texture.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace Common {

void FlipRGBA8Texture(std::span<u8> tex, u32 width, u32 height) {
    ASSERT(tex.size() == width * height * 4);
    const u32 line_size = width * 4;
    for (u32 line = 0; line < height / 2; line++) {
        u8* ptr1 = tex.data() + line * line_size;
        u8* ptr2 = tex.data() + (height - line - 1) * line_size;
#if defined(__ARM_NEON)
        u32 offset = 0;
        for (; offset + 64 <= line_size; offset += 64) {
            uint8x16_t a0 = vld1q_u8(ptr1 + offset);
            uint8x16_t a1 = vld1q_u8(ptr1 + offset + 16);
            uint8x16_t a2 = vld1q_u8(ptr1 + offset + 32);
            uint8x16_t a3 = vld1q_u8(ptr1 + offset + 48);
            uint8x16_t b0 = vld1q_u8(ptr2 + offset);
            uint8x16_t b1 = vld1q_u8(ptr2 + offset + 16);
            uint8x16_t b2 = vld1q_u8(ptr2 + offset + 32);
            uint8x16_t b3 = vld1q_u8(ptr2 + offset + 48);
            vst1q_u8(ptr1 + offset, b0);
            vst1q_u8(ptr1 + offset + 16, b1);
            vst1q_u8(ptr1 + offset + 32, b2);
            vst1q_u8(ptr1 + offset + 48, b3);
            vst1q_u8(ptr2 + offset, a0);
            vst1q_u8(ptr2 + offset + 16, a1);
            vst1q_u8(ptr2 + offset + 32, a2);
            vst1q_u8(ptr2 + offset + 48, a3);
        }
        for (; offset + 16 <= line_size; offset += 16) {
            uint8x16_t a = vld1q_u8(ptr1 + offset);
            uint8x16_t b = vld1q_u8(ptr2 + offset);
            vst1q_u8(ptr1 + offset, b);
            vst1q_u8(ptr2 + offset, a);
        }
        for (; offset < line_size; offset++) {
            std::swap(ptr1[offset], ptr2[offset]);
        }
#else
        std::swap_ranges(ptr1, ptr1 + line_size, ptr2);
#endif
    }
}

} // namespace Common
