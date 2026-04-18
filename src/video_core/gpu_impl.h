// Copyright 2023 Citra Emulator Project
// Copyright 2024 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include "common/archives.h"
#include "common/microprofile.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/service/gsp/gsp_gpu.h"
#include "core/hle/service/plgldr/plgldr.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/gpu.h"
#include "video_core/gpu_debugger.h"
#include "video_core/gpu_impl.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/regs_lcd.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_software/sw_blitter.h"
#include "video_core/right_eye_disabler.h"
#include "video_core/video_core.h"

namespace VideoCore {
struct GPU::Impl {
    Core::Timing& timing;
    Core::System& system;
    Memory::MemorySystem& memory;
    std::shared_ptr<Pica::DebugContext> debug_context;
    Pica::PicaCore pica;
    GraphicsDebugger gpu_debugger;
    std::unique_ptr<RendererBase> renderer;
    RasterizerInterface* rasterizer;
    std::unique_ptr<SwRenderer::SwBlitter> sw_blitter;
    u64 current_program_id{};
    Core::TimingEventType* vblank_event;
    Service::GSP::InterruptHandler signal_interrupt;

    std::thread gpu_thread;
    std::mutex gpu_mutex;
    std::condition_variable gpu_work_cv;
    std::condition_variable gpu_idle_cv;
    std::queue<std::function<void()>> gpu_queue;
    bool gpu_running{true};
    int gpu_pending{0};

    explicit Impl(Core::System& system, Frontend::EmuWindow& emu_window,
                  Frontend::EmuWindow* secondary_window)
        : timing{system.CoreTiming()}, system{system}, memory{system.Memory()},
          debug_context{Pica::g_debug_context}, pica{memory, debug_context},
          renderer{VideoCore::CreateRenderer(emu_window, secondary_window, pica, system)},
          rasterizer{renderer->Rasterizer()},
          sw_blitter{std::make_unique<SwRenderer::SwBlitter>(memory, rasterizer)} {
        gpu_thread = std::thread([this] {
            Common::SetCurrentThreadName("GPUThread");
            Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock lock(gpu_mutex);
                    gpu_work_cv.wait(lock, [this] {
                        return !gpu_queue.empty() || !gpu_running;
                    });
                    if (!gpu_running && gpu_queue.empty()) {
                        return;
                    }
                    task = std::move(gpu_queue.front());
                    gpu_queue.pop();
                }
                task();
                {
                    std::lock_guard lock(gpu_mutex);
                    gpu_pending--;
                }
                gpu_idle_cv.notify_all();
            }
        });
    }

    ~Impl() {
        {
            std::lock_guard lock(gpu_mutex);
            gpu_running = false;
        }
        gpu_work_cv.notify_one();
        if (gpu_thread.joinable()) {
            gpu_thread.join();
        }
    }

    void EnqueueGPUWork(std::function<void()> work) {
        {
            std::lock_guard lock(gpu_mutex);
            gpu_pending++;
            gpu_queue.push(std::move(work));
        }
        gpu_work_cv.notify_one();
    }

    void SyncGPU() {
        std::unique_lock lock(gpu_mutex);
        gpu_idle_cv.wait(lock, [this] { return gpu_pending == 0; });
    }
};
} // namespace VideoCore
