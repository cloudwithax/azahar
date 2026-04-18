#include "common/arch.h"

#if CITRA_ARCH(arm64) && defined(__linux__)

#include "core/memory_fastmem.h"

#include <cerrno>
#include <cstring>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "common/alignment.h"
#include "common/logging/log.h"
#include "core/memory.h"

namespace Memory {

static int CreateMemfd(const char* name, size_t size) {
    int fd = static_cast<int>(syscall(SYS_memfd_create, name, MFD_CLOEXEC));
    if (fd < 0) {
        return -1;
    }
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

FastMemManager::FastMemManager() {
    fastmem_base = static_cast<u8*>(
        mmap(nullptr, FASTMEM_SIZE, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE, -1, 0));
    if (fastmem_base == MAP_FAILED) {
        LOG_ERROR(HW_Memory, "Failed to allocate fastmem region ({}): {}", errno, strerror(errno));
        fastmem_base = nullptr;
        return;
    }
    LOG_INFO(HW_Memory, "Fastmem region allocated at {} (size 4GB)",
             reinterpret_cast<void*>(fastmem_base));
}

FastMemManager::~FastMemManager() {
    if (fastmem_base) {
        munmap(fastmem_base, FASTMEM_SIZE);
    }
    for (auto* region : {&fcram_region, &vram_region, &dsp_region, &n3ds_region}) {
        if (region->fd >= 0) {
            close(region->fd);
        }
    }
}

void FastMemManager::SetBackingMemory(u8* fcram, size_t fcram_size, u8* vram, size_t vram_size,
                                      u8* dsp_ram, size_t dsp_size, u8* n3ds_ram,
                                      size_t n3ds_size) {
    if (!fastmem_base) return;

    auto setup = [this](MemfdRegion& region, const char* name, u8* backing, size_t size) {
        int fd = CreateMemfd(name, size);
        if (fd < 0) {
            LOG_WARNING(HW_Memory, "Failed to create memfd for {}: {}", name, strerror(errno));
            region.fd = -1;
            return;
        }

        auto* mapped = mmap(backing, size, PROT_READ | PROT_WRITE,
                            MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 0);
        if (mapped == MAP_FAILED) {
            LOG_WARNING(HW_Memory, "memfd remap for {} failed, trying MAP_FIXED: {}",
                        name, strerror(errno));
            mapped = mmap(backing, size, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_FIXED, fd, 0);
        }
        if (mapped == MAP_FAILED) {
            LOG_WARNING(HW_Memory, "memfd remap for {} failed entirely: {}", name, strerror(errno));
            close(fd);
            region.fd = -1;
            return;
        }

        region.fd = fd;
        region.base = backing;
        region.size = size;
        LOG_INFO(HW_Memory, "Fastmem memfd for {} at {} ({} MB, fd={})",
                 name, reinterpret_cast<void*>(backing), size / 1024 / 1024, fd);
    };

    setup(fcram_region, "azahar_fcram", fcram, fcram_size);
    setup(vram_region, "azahar_vram", vram, vram_size);
    setup(dsp_region, "azahar_dsp", dsp_ram, dsp_size);
    setup(n3ds_region, "azahar_n3ds", n3ds_ram, n3ds_size);
}

int FastMemManager::FindMemfdForPtr(const u8* ptr, size_t& offset_out) const {
    auto check = [&](const MemfdRegion& region) {
        if (region.fd >= 0 && ptr >= region.base && ptr < region.base + region.size) {
            offset_out = static_cast<size_t>(ptr - region.base);
            return true;
        }
        return false;
    };
    if (check(fcram_region)) return fcram_region.fd;
    if (check(vram_region)) return vram_region.fd;
    if (check(dsp_region)) return dsp_region.fd;
    if (check(n3ds_region)) return n3ds_region.fd;
    return -1;
}

void FastMemManager::MapRegion(VAddr base, u32 size, const u8* backing_ptr) {
    if (!fastmem_base) return;

    const size_t page_count = size / CITRA_PAGE_SIZE;
    for (size_t i = 0; i < page_count; i++) {
        const u8* page_backing = backing_ptr ? backing_ptr + i * CITRA_PAGE_SIZE : nullptr;
        size_t fd_offset = 0;
        int fd = page_backing ? FindMemfdForPtr(page_backing, fd_offset) : -1;

        u8* dst = fastmem_base + base + i * CITRA_PAGE_SIZE;

        if (fd >= 0) {
            auto* result = mmap(dst, CITRA_PAGE_SIZE, PROT_READ | PROT_WRITE,
                                MAP_SHARED | MAP_FIXED, fd, static_cast<off_t>(fd_offset));
            if (result == MAP_FAILED) {
                LOG_ERROR(HW_Memory, "Fastmem shared map failed @ {:08X}: {}",
                          base + i * CITRA_PAGE_SIZE, strerror(errno));
            }
        } else {
            // No memfd for this backing page — leave as PROT_NONE.
            // Dynarmic's fastmem fault handler will catch the SIGSEGV
            // and fall back to page table lookup. This is correct;
            // the old MAP_ANONYMOUS|MAP_PRIVATE fallback was broken
            // because CoW pages don't propagate writes back to the
            // real backing memory.
        }
    }
}

void FastMemManager::UnmapRegion(VAddr base, u32 size) {
    if (!fastmem_base) return;

    const size_t page_count = size / CITRA_PAGE_SIZE;
    for (size_t i = 0; i < page_count; i++) {
        mprotect(fastmem_base + base + i * CITRA_PAGE_SIZE, CITRA_PAGE_SIZE, PROT_NONE);
    }
}

void FastMemManager::SetRasterizerCached(VAddr base, u32 size, bool cached) {
    if (!fastmem_base) return;

    const size_t page_count = Common::AlignUp(size, CITRA_PAGE_SIZE) / CITRA_PAGE_SIZE;
    const VAddr aligned_base = Common::AlignDown(base, CITRA_PAGE_SIZE);

    for (size_t i = 0; i < page_count; i++) {
        mprotect(fastmem_base + aligned_base + i * CITRA_PAGE_SIZE, CITRA_PAGE_SIZE,
                 cached ? PROT_NONE : (PROT_READ | PROT_WRITE));
    }
}

void FastMemManager::ReprotectPage(VAddr page_addr, bool writable) {
    if (!fastmem_base) return;
    mprotect(fastmem_base + page_addr, CITRA_PAGE_SIZE,
             writable ? (PROT_READ | PROT_WRITE) : PROT_NONE);
}

} // namespace Memory

#else

#include "core/memory_fastmem.h"

namespace Memory {

FastMemManager::FastMemManager() = default;
FastMemManager::~FastMemManager() = default;
void FastMemManager::SetBackingMemory(u8*, size_t, u8*, size_t, u8*, size_t, u8*, size_t) {}
void FastMemManager::MapRegion(VAddr, u32, const u8*) {}
void FastMemManager::UnmapRegion(VAddr, u32) {}
void FastMemManager::SetRasterizerCached(VAddr, u32, bool) {}
void FastMemManager::ReprotectPage(VAddr, bool) {}

} // namespace Memory

#endif
