// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/frontend/emu_window.h"
#include "core/tracer/recorder.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/renderer_base.h"

namespace VideoCore {

RendererBase::RendererBase(Core::System& system_, Frontend::EmuWindow& window,
                           Frontend::EmuWindow* secondary_window_)
    : system{system_}, render_window{window}, secondary_window{secondary_window_} {}

RendererBase::~RendererBase() = default;

u32 RendererBase::GetRequestedResolutionScaleFactor() const {
    const auto graphics_api = Settings::values.graphics_api.GetValue();
    if (graphics_api == Settings::GraphicsAPI::Software) {
        // Software renderer always render at native resolution
        return 1;
    }

    const u32 scale_factor = Settings::values.resolution_factor.GetValue();
    return scale_factor != 0 ? scale_factor
                             : render_window.GetFramebufferLayout().GetScalingRatio();
}

u32 RendererBase::GetResolutionScaleFactor() {
    const u32 requested_scale = GetRequestedResolutionScaleFactor();
#if defined(ANDROID)
    if (Settings::values.graphics_api.GetValue() == Settings::GraphicsAPI::Vulkan &&
        adaptive_resolution_cap != 0) {
        return std::min(requested_scale, adaptive_resolution_cap);
    }
#endif
    return requested_scale;
}

void RendererBase::ResetAdaptivePerformanceControls() {
#if defined(ANDROID)
    adaptive_resolution_cap = 0;
    slow_frame_streak = 0;
    fast_frame_streak = 0;
    adaptive_quality_slow_streak = 0;
    adaptive_quality_fast_streak = 0;
    last_effective_resolution_scale = GetRequestedResolutionScaleFactor();
    ApplyAdaptiveQualityLevel(0);
#endif
}

u32 RendererBase::GetAdaptiveQualityMaxLevel() const {
    u32 max_level = 0;
    if (adaptive_requested_custom_textures) {
        max_level++;
    }
    if (adaptive_requested_texture_filter != Settings::TextureFilter::NoFilter) {
        max_level++;
    }
    if (adaptive_requested_shaders_accurate_mul) {
        max_level++;
    }
    return max_level;
}

void RendererBase::ApplyAdaptiveQualityLevel(u32 level) {
#if defined(ANDROID)
    const u32 clamped_level = std::min(level, GetAdaptiveQualityMaxLevel());
    u32 remaining = clamped_level;

    const bool disable_custom_textures = adaptive_requested_custom_textures && remaining-- > 0;
    const bool disable_texture_filter =
        adaptive_requested_texture_filter != Settings::TextureFilter::NoFilter && remaining-- > 0;
    const bool disable_accurate_mul = adaptive_requested_shaders_accurate_mul && remaining-- > 0;

    const bool target_custom_textures =
        disable_custom_textures ? false : adaptive_requested_custom_textures;
    const auto target_texture_filter = disable_texture_filter
                                           ? Settings::TextureFilter::NoFilter
                                           : adaptive_requested_texture_filter;
    const bool target_accurate_mul =
        disable_accurate_mul ? false : adaptive_requested_shaders_accurate_mul;

    const bool previous_accurate_mul = Settings::values.shaders_accurate_mul.GetValue();
    const auto previous_texture_filter = Settings::values.texture_filter.GetValue();
    const bool previous_custom_textures = Settings::values.custom_textures.GetValue();

    if (previous_accurate_mul != target_accurate_mul) {
        Settings::values.shaders_accurate_mul.SetValue(target_accurate_mul);
        settings.shader_update_requested = true;
    }
    if (previous_texture_filter != target_texture_filter) {
        Settings::values.texture_filter.SetValue(target_texture_filter);
    }
    if (previous_custom_textures != target_custom_textures) {
        Settings::values.custom_textures.SetValue(target_custom_textures);
    }

    if (adaptive_quality_level == clamped_level) {
        return;
    }

    adaptive_quality_level = clamped_level;
    if (clamped_level == 0) {
        LOG_INFO(Render, "Adaptive quality restored runtime renderer settings");
    } else {
        LOG_INFO(Render,
                 "Adaptive quality engaged level {} (custom_textures={}, texture_filter={}, accurate_mul={})",
                 clamped_level, target_custom_textures, static_cast<u32>(target_texture_filter),
                 target_accurate_mul);
    }
#else
    static_cast<void>(level);
#endif
}

void RendererBase::UpdateAdaptiveQualityControls(double frame_scale, u32 requested_scale,
                                                 u32 effective_scale) {
#if defined(ANDROID)
    if (adaptive_quality_level == 0) {
        adaptive_requested_shaders_accurate_mul = Settings::values.shaders_accurate_mul.GetValue();
        adaptive_requested_texture_filter = Settings::values.texture_filter.GetValue();
        adaptive_requested_custom_textures = Settings::values.custom_textures.GetValue();
    }

    const u32 max_quality_level = GetAdaptiveQualityMaxLevel();
    if (max_quality_level == 0) {
        adaptive_quality_slow_streak = 0;
        adaptive_quality_fast_streak = 0;
        ApplyAdaptiveQualityLevel(0);
        return;
    }

    const bool at_native_floor = effective_scale <= 1;
    constexpr double QualityOverloadThreshold = 1.08;
    constexpr double QualityRecoveryThreshold = 0.90;
    constexpr u32 SlowFramesToShedQuality = 75;
    constexpr u32 FastFramesToRestoreQuality = 300;

    if (at_native_floor && frame_scale > QualityOverloadThreshold) {
        adaptive_quality_slow_streak =
            std::min(adaptive_quality_slow_streak + 1, SlowFramesToShedQuality);
        adaptive_quality_fast_streak = 0;
    } else if (frame_scale < QualityRecoveryThreshold) {
        adaptive_quality_fast_streak =
            std::min(adaptive_quality_fast_streak + 1, FastFramesToRestoreQuality);
        adaptive_quality_slow_streak = 0;
    } else {
        adaptive_quality_slow_streak = 0;
        adaptive_quality_fast_streak = 0;
    }

    if (adaptive_quality_slow_streak >= SlowFramesToShedQuality &&
        adaptive_quality_level < max_quality_level) {
        adaptive_quality_slow_streak = 0;
        ApplyAdaptiveQualityLevel(adaptive_quality_level + 1);
    } else if (adaptive_quality_fast_streak >= FastFramesToRestoreQuality &&
               adaptive_quality_level > 0) {
        adaptive_quality_fast_streak = 0;
        ApplyAdaptiveQualityLevel(adaptive_quality_level - 1);
    }
#else
    static_cast<void>(frame_scale);
    static_cast<void>(effective_scale);
#endif
    static_cast<void>(requested_scale);
}

void RendererBase::UpdateAdaptiveResolutionScale() {
    const double frame_scale = system.perf_stats->GetStableFrameTimeScale();
    current_fps = frame_scale > 0.0 ? static_cast<f32>(SCREEN_REFRESH_RATE / frame_scale) : 0.0f;

#if defined(ANDROID)
    if (Settings::values.graphics_api.GetValue() != Settings::GraphicsAPI::Vulkan) {
        ResetAdaptivePerformanceControls();
        return;
    }

    const u32 requested_scale = GetRequestedResolutionScaleFactor();
    adaptive_resolution_cap =
        adaptive_resolution_cap == 0 ? requested_scale : adaptive_resolution_cap;
    adaptive_resolution_cap = std::min(adaptive_resolution_cap, requested_scale);

    if (requested_scale <= 1) {
        adaptive_resolution_cap = requested_scale;
        slow_frame_streak = 0;
        fast_frame_streak = 0;
    } else {
        constexpr double OverloadThreshold = 1.12;
        constexpr double RecoveryThreshold = 0.92;
        constexpr u32 SlowFramesToStepDown = 45;
        constexpr u32 FastFramesToStepUp = 240;

        if (frame_scale > OverloadThreshold) {
            slow_frame_streak = std::min(slow_frame_streak + 1, SlowFramesToStepDown);
            fast_frame_streak = 0;
        } else if (frame_scale < RecoveryThreshold) {
            fast_frame_streak = std::min(fast_frame_streak + 1, FastFramesToStepUp);
            slow_frame_streak = 0;
        } else {
            slow_frame_streak = 0;
            fast_frame_streak = 0;
        }

        const u32 previous_scale = std::min(requested_scale, adaptive_resolution_cap);
        if (slow_frame_streak >= SlowFramesToStepDown && adaptive_resolution_cap > 1) {
            adaptive_resolution_cap--;
            slow_frame_streak = 0;
            LOG_INFO(Render, "Adaptive resolution reduced Vulkan internal scale from {}x to {}x",
                     previous_scale, adaptive_resolution_cap);
        } else if (fast_frame_streak >= FastFramesToStepUp && adaptive_resolution_cap < requested_scale &&
                   adaptive_quality_level == 0) {
            adaptive_resolution_cap++;
            fast_frame_streak = 0;
            LOG_INFO(Render, "Adaptive resolution restored Vulkan internal scale from {}x to {}x",
                     previous_scale, adaptive_resolution_cap);
        }
    }

    last_effective_resolution_scale = std::min(requested_scale, adaptive_resolution_cap);
    UpdateAdaptiveQualityControls(frame_scale, requested_scale, last_effective_resolution_scale);
#endif
}

void RendererBase::UpdateCurrentFramebufferLayout(bool is_portrait_mode) {
    const auto update_layout = [is_portrait_mode](Frontend::EmuWindow& window) {
        const Layout::FramebufferLayout& layout = window.GetFramebufferLayout();
        window.UpdateCurrentFramebufferLayout(layout.width, layout.height, is_portrait_mode);
    };
    update_layout(render_window);
    if (secondary_window != nullptr) {
        update_layout(*secondary_window);
    }
}

void RendererBase::EndFrame() {
    current_frame++;

    system.perf_stats->EndSystemFrame();
    UpdateAdaptiveResolutionScale();

    render_window.PollEvents();

    system.frame_limiter.DoFrameLimiting(system.CoreTiming().GetGlobalTimeUs());
    system.perf_stats->BeginSystemFrame();
}

bool RendererBase::IsScreenshotPending() const {
    return settings.screenshot_requested;
}

void RendererBase::RequestScreenshot(void* data, std::function<void(bool)> callback,
                                     const Layout::FramebufferLayout& layout) {
    if (settings.screenshot_requested) {
        LOG_ERROR(Render, "A screenshot is already requested or in progress, ignoring the request");
        return;
    }
    settings.screenshot_bits = data;
    settings.screenshot_complete_callback = callback;
    settings.screenshot_framebuffer_layout = layout;
    settings.screenshot_requested = true;
}
} // namespace VideoCore
