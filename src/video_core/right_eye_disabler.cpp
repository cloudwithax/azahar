// Copyright 2024 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/hacks/hack_manager.h"
#include "common/settings.h"
#include "right_eye_disabler.h"
#include "video_core/gpu.h"
#include "video_core/gpu_impl.h"

namespace VideoCore {
bool RightEyeDisabler::ShouldAllowCmdQueueTrigger(PAddr addr, u32 size) {
    if (!enabled || !enable_for_frame)
        return true;

    constexpr u32 top_screen_size = 0x00469000;

    if (report_end_frame_pending) {
        ReportEndFrame();
        report_end_frame_pending = false;
    }
    cmd_queue_trigger_happened = true;

    auto guess = gpu.impl->pica.GuessCmdRenderProperties(addr, size);
    if (guess.vp_height == top_screen_size && !top_screen_blocked) {
        if (top_screen_buf == 0) {
            top_screen_buf = guess.paddr;
        }
        top_screen_drawn = true;
        if (top_screen_transfered) {
            cmd_trigger_blocked = true;
            return false;
        }
    }

    cmd_trigger_blocked = false;
    return true;
}
bool RightEyeDisabler::ShouldAllowDisplayTransfer(PAddr src_address, size_t size) {
    if (!enabled || !enable_for_frame)
        return true;

    if (size >= 400 && !top_screen_blocked) {
        if (top_screen_drawn && src_address == top_screen_buf) {
            top_screen_transfered = true;
        }

        if (src_address == top_screen_buf && cmd_trigger_blocked) {
            top_screen_blocked = true;
            return false;
        }
    }

    if (cmd_queue_trigger_happened)
        display_tranfer_happened = true;
    return true;
}
void RightEyeDisabler::ReportEndFrame() {
    if (!enabled)
        return;

    constexpr double OverloadThreshold = 1.08;
    constexpr double RecoveryThreshold = 0.95;
    constexpr u32 OverloadFrames = 30;
    constexpr u32 RecoveryFrames = 180;

    const double frame_scale = gpu.impl->system.perf_stats->GetStableFrameTimeScale();
    if (frame_scale > OverloadThreshold) {
        overload_streak = std::min(overload_streak + 1, OverloadFrames);
        recovery_streak = 0;
    } else if (frame_scale < RecoveryThreshold) {
        recovery_streak = std::min(recovery_streak + 1, RecoveryFrames);
        overload_streak = 0;
    } else {
        overload_streak = 0;
        recovery_streak = 0;
    }

    if (overload_streak >= OverloadFrames && !adaptive_disable_active) {
        adaptive_disable_active = true;
        overload_streak = 0;
        LOG_INFO(Render, "Adaptive stereo shedding enabled right-eye suppression under GPU load");
    } else if (recovery_streak >= RecoveryFrames && adaptive_disable_active) {
        adaptive_disable_active = false;
        recovery_streak = 0;
        LOG_INFO(Render, "Adaptive stereo shedding restored right-eye rendering after recovery");
    }

    const bool forced_disable = Common::Hacks::hack_manager.OverrideBooleanSetting(
        Common::Hacks::HackType::RIGHT_EYE_DISABLE, gpu.impl->current_program_id,
        Settings::values.disable_right_eye_render.GetValue());
    enable_for_frame = forced_disable || adaptive_disable_active;

    if (display_tranfer_happened) {
        top_screen_drawn = false;
        top_screen_transfered = false;
        top_screen_blocked = false;
        cmd_queue_trigger_happened = false;
        cmd_trigger_blocked = false;
        display_tranfer_happened = false;
        top_screen_buf = 0;
    } else {
        report_end_frame_pending = true;
    }
}
} // namespace VideoCore
