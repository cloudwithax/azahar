// Copyright 2013 Dolphin Emulator Project / 2022 Citra Emulator Project
// Licensed under GPLv2+
// Refer to the license.txt file included.

#include "common/arch.h"
#if CITRA_ARCH(arm64)

#include <cstring>
#include <fstream>
#include <string>

#ifdef __APPLE__
// clang-format off
#include <sys/types.h>
#include <sys/sysctl.h>
// clang-format on
#elif !defined(_WIN32)
#ifndef __FreeBSD__
#include <asm/hwcap.h>
#endif // __FreeBSD__
#include <sys/auxv.h>
#include <unistd.h>
#endif // __APPLE__

#include "common/aarch64/cpu_detect.h"
#include "common/file_util.h"

namespace Common {

#ifdef __APPLE__
static std::string GetCPUString() {
    char buf[128];
    std::size_t buf_len = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", &buf, &buf_len, NULL, 0) == -1) {
        return "Unknown";
    }
    return buf;
}
#elif !defined(WIN32)
static std::string GetCPUString() {
    constexpr char procfile[] = "/proc/cpuinfo";
    constexpr char marker[] = "Hardware\t: ";
    std::string cpu_string = "Unknown";

    std::string line;
    std::ifstream file;
    OpenFStream(file, procfile, std::ios_base::in);

    if (!file)
        return cpu_string;

    while (std::getline(file, line)) {
        if (line.find(marker) != std::string::npos) {
            cpu_string = line.substr(strlen(marker));
            break;
        }
    }

    return cpu_string;
}
#endif // __APPLE__

// Detects the various CPU features
static CPUCaps Detect() {
    CPUCaps caps;
    // Set some defaults here
    caps.fma = true;
    caps.afp = false;
    caps.dotprod = false;
    caps.fhm = false;
    caps.fp16 = false;
    caps.sha3 = false;
    caps.sha512 = false;

#ifdef __APPLE__
    // M-series CPUs have all of these
    caps.fp = true;
    caps.asimd = true;
    caps.aes = true;
    caps.crc32 = true;
    caps.sha1 = true;
    caps.sha2 = true;
    caps.dotprod = true;
    caps.fhm = true;
    caps.fp16 = true;
    caps.sha3 = true;
    caps.sha512 = true;
    caps.cpu_string = GetCPUString();
#elif defined(_WIN32)
    // Windows does not provide any mechanism for querying the system registers on ARMv8, unlike
    // Linux which traps the register reads and emulates them in the kernel. There are environment
    // variables containing some of the CPU-specific values, which we could use for a lookup table
    // in the future. For now, assume all features are present as all known devices which are
    // Windows-on-ARM compatible also support these extensions.
    caps.fp = true;
    caps.asimd = true;
    caps.aes = true;
    caps.crc32 = true;
    caps.sha1 = true;
    caps.sha2 = true;
    caps.dotprod = true;
    caps.fhm = true;
    caps.fp16 = true;
    caps.sha3 = true;
    caps.sha512 = true;
#else
    caps.cpu_string = GetCPUString();

#ifdef __FreeBSD__
    u_long hwcaps = 0;
    u_long hwcaps2 = 0;
    elf_aux_info(AT_HWCAP, &hwcaps, sizeof(u_long));
    elf_aux_info(AT_HWCAP2, &hwcaps2, sizeof(u_long));
#else
    unsigned long hwcaps = getauxval(AT_HWCAP);
    unsigned long hwcaps2 = 0;
#ifdef AT_HWCAP2
    hwcaps2 = getauxval(AT_HWCAP2);
#endif
#endif // __FreeBSD__
    caps.fp = hwcaps & HWCAP_FP;
    caps.asimd = hwcaps & HWCAP_ASIMD;
    caps.aes = hwcaps & HWCAP_AES;
    caps.crc32 = hwcaps & HWCAP_CRC32;
    caps.sha1 = hwcaps & HWCAP_SHA1;
    caps.sha2 = hwcaps & HWCAP_SHA2;
    caps.sha3 = hwcaps & HWCAP_SHA3;
    caps.sha512 = hwcaps2 & (1 << 21); // HWCAP2_SHA512
    caps.dotprod = hwcaps2 & (1 << 13); // HWCAP2_ASIMDDP
    caps.fhm = hwcaps2 & (1 << 23); // HWCAP2_ASIMDFHM
    caps.fp16 = hwcaps & HWCAP_FPHP && hwcaps & HWCAP_ASIMDHP;
#endif // __APPLE__
    return caps;
}

const CPUCaps& GetCPUCaps() {
    static CPUCaps caps = Detect();
    return caps;
}

} // namespace Common

#endif // CITRA_ARCH(arm64)
