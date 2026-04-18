#pragma once

#include <cstdint>
#include <optional>
#include "common/common_types.h"

namespace Memory {

class FastMemManager {
public:
    FastMemManager();
    ~FastMemManager();

    FastMemManager(const FastMemManager&) = delete;
    FastMemManager& operator=(const FastMemManager&) = delete;

    bool IsEnabled() const {
        return fastmem_base != nullptr;
    }

    std::optional<uintptr_t> GetFastmemBase() const {
        if (fastmem_base) {
            return reinterpret_cast<uintptr_t>(fastmem_base);
        }
        return std::nullopt;
    }

    void SetBackingMemory(u8* fcram, size_t fcram_size, u8* vram, size_t vram_size,
                          u8* dsp_ram, size_t dsp_size, u8* n3ds_ram, size_t n3ds_size);

    void MapRegion(VAddr base, u32 size, const u8* backing_ptr);
    void UnmapRegion(VAddr base, u32 size);
    void SetRasterizerCached(VAddr base, u32 size, bool cached);
    void ReprotectPage(VAddr page_addr, bool writable);

private:
    struct MemfdRegion {
        int fd = -1;
        u8* base = nullptr;
        size_t size = 0;
    };

    int FindMemfdForPtr(const u8* ptr, size_t& offset_out) const;

    u8* fastmem_base = nullptr;
    static constexpr size_t FASTMEM_SIZE = 1ULL << 32;

    MemfdRegion fcram_region;
    MemfdRegion vram_region;
    MemfdRegion dsp_region;
    MemfdRegion n3ds_region;
};

} // namespace Memory
