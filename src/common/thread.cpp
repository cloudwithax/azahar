// Copyright 2013 Dolphin Emulator Project / 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <string>

#include "common/error.h"
#include "common/logging/log.h"
#include "common/thread.h"
#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(_WIN32)
#include <windows.h>
#include "common/string_util.h"
#else
#if defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
#include <pthread_np.h>
#else
#include <pthread.h>
#endif
#include <sched.h>
#include <sys/resource.h>
#ifdef __ANDROID__
#include <sys/syscall.h>
#endif
#endif
#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __FreeBSD__
#define cpu_set_t cpuset_t
#endif

#ifdef __ANDROID__
static int GetMaxBigCore() {
    FILE* fp = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r");
    if (!fp) {
        return -1;
    }
    long max_freq = 0;
    fscanf(fp, "%ld", &max_freq);
    fclose(fp);

    int best_core = -1;
    long best_freq = 0;
    for (int i = 0; i < 8; i++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        fp = fopen(path, "r");
        if (!fp) {
            break;
        }
        long freq = 0;
        fscanf(fp, "%ld", &freq);
        fclose(fp);
        if (freq > best_freq) {
            best_freq = freq;
            best_core = i;
        }
    }
    return best_core;
}
#endif

namespace Common {

#ifdef _WIN32

void SetCurrentThreadPriority(ThreadPriority new_priority) {
    auto handle = GetCurrentThread();
    int windows_priority = 0;
    switch (new_priority) {
    case ThreadPriority::Low:
        windows_priority = THREAD_PRIORITY_BELOW_NORMAL;
        break;
    case ThreadPriority::Normal:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    case ThreadPriority::High:
        windows_priority = THREAD_PRIORITY_ABOVE_NORMAL;
        break;
    case ThreadPriority::VeryHigh:
        windows_priority = THREAD_PRIORITY_HIGHEST;
        break;
    case ThreadPriority::Critical:
        windows_priority = THREAD_PRIORITY_TIME_CRITICAL;
        break;
    default:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    }
    SetThreadPriority(handle, windows_priority);
}

#else

namespace {

int GetNiceValue(ThreadPriority priority) {
    switch (priority) {
    case ThreadPriority::Low:
        return 10;
    case ThreadPriority::Normal:
        return 0;
    case ThreadPriority::High:
        return -4;
    case ThreadPriority::VeryHigh:
        return -8;
    case ThreadPriority::Critical:
        return -10;
    default:
        return 0;
    }
}

} // Anonymous namespace

void SetCurrentThreadPriority(ThreadPriority new_priority) {
#ifdef __ANDROID__
    // On Android, unprivileged apps cannot set negative nice values via setpriority().
    // Use the Android-specific thread priority range with the thread's kernel tid.
    // Android priority range: 19 (lowest) to -20 (highest, requires permission).
    // For unprivileged apps the usable range is roughly 19 to -8.
    // ANDROID_PRIORITY_AUDIO = -16, ANDROID_PRIORITY_URGENT_AUDIO = -19
    // ANDROID_PRIORITY_DISPLAY = -4, ANDROID_PRIORITY_FOREGROUND = -2
    int android_priority;
    switch (new_priority) {
    case ThreadPriority::Low:
        android_priority = 10; // ANDROID_PRIORITY_BACKGROUND
        break;
    case ThreadPriority::Normal:
        android_priority = 0; // ANDROID_PRIORITY_DEFAULT
        break;
    case ThreadPriority::High:
        android_priority = -4; // ANDROID_PRIORITY_DISPLAY
        break;
    case ThreadPriority::VeryHigh:
        android_priority = -8; // ANDROID_PRIORITY_URGENT_DISPLAY
        break;
    case ThreadPriority::Critical:
        android_priority = -16; // ANDROID_PRIORITY_AUDIO
        break;
    default:
        android_priority = 0;
        break;
    }
    // Use the thread id (not process id) to set per-thread priority on Android.
    const pid_t tid = static_cast<pid_t>(syscall(__NR_gettid));
    if (setpriority(PRIO_PROCESS, tid, android_priority) != 0) {
        LOG_WARNING(Common, "Failed to set Android thread priority {} for tid {}: {}",
                    android_priority, tid, GetLastErrorMsg());
    }
#else
    const int nice_value = GetNiceValue(new_priority);
    if (setpriority(PRIO_PROCESS, 0, nice_value) == 0) {
        return;
    }

    pthread_t this_thread = pthread_self();
    const auto scheduling_type = SCHED_OTHER;
    s32 max_prio = sched_get_priority_max(scheduling_type);
    s32 min_prio = sched_get_priority_min(scheduling_type);
    u32 level = std::max(static_cast<u32>(new_priority) + 1, 4U);

    struct sched_param params;
    if (max_prio > min_prio) {
        params.sched_priority = min_prio + ((max_prio - min_prio) * level) / 4;
    } else {
        params.sched_priority = min_prio - ((min_prio - max_prio) * level) / 4;
    }

    pthread_setschedparam(this_thread, scheduling_type, &params);
#endif
}

#endif

#ifdef _MSC_VER

// Sets the debugger-visible name of the current thread.
void SetCurrentThreadName(const char* name) {
    SetThreadDescription(GetCurrentThread(), UTF8ToUTF16W(name).data());
}

#else // !MSVC_VER, so must be POSIX threads

// MinGW with the POSIX threading model does not support pthread_setname_np
#if !defined(_WIN32) || defined(_MSC_VER)
void SetCurrentThreadName(const char* name) {
#ifdef __APPLE__
    pthread_setname_np(name);
#elif defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    pthread_set_name_np(pthread_self(), name);
#elif defined(__NetBSD__)
    pthread_setname_np(pthread_self(), "%s", (void*)name);
#elif defined(__linux__)
    // Linux limits thread names to 15 characters and will outright reject any
    // attempt to set a longer name with ERANGE.
    std::string truncated(name, std::min(strlen(name), static_cast<std::size_t>(15)));
    if (int e = pthread_setname_np(pthread_self(), truncated.c_str())) {
        errno = e;
        LOG_ERROR(Common, "Failed to set thread name to '{}': {}", truncated, GetLastErrorMsg());
    }
#else
    pthread_setname_np(pthread_self(), name);
#endif
}
#endif

#if defined(_WIN32)
void SetCurrentThreadName(const char*) {
    // Do Nothing on MingW
}
#endif

#endif

#ifdef __ANDROID__
void SetThreadAffinityBigCores() {
    int best_core = GetMaxBigCore();
    if (best_core < 0) {
        return;
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);

    long best_freq = 0;
    for (int i = 0; i < 8; i++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        FILE* fp = fopen(path, "r");
        if (!fp) {
            break;
        }
        long freq = 0;
        fscanf(fp, "%ld", &freq);
        fclose(fp);
        if (freq >= best_freq) {
            CPU_SET(i, &mask);
        }
    }

    if (CPU_COUNT(&mask) > 0) {
        sched_setaffinity(static_cast<pid_t>(syscall(__NR_gettid)), sizeof(mask), &mask);
    }
}
#endif

} // namespace Common
