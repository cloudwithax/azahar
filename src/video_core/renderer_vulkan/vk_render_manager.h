// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <mutex>

#include "common/math_util.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
enum class PixelFormat : u32;
}

namespace Vulkan {

class Instance;
class Scheduler;
class Framebuffer;

struct RenderPass {
    vk::Framebuffer framebuffer;
    vk::RenderPass render_pass;
    vk::Rect2D render_area;
    vk::ClearValue clear;
    u32 do_clear;

    bool operator==(const RenderPass& other) const noexcept {
        return std::tie(framebuffer, render_pass, render_area, do_clear) ==
                   std::tie(other.framebuffer, other.render_pass, other.render_area,
                            other.do_clear) &&
               std::memcmp(&clear, &other.clear, sizeof(vk::ClearValue)) == 0;
    }
};

struct RenderHints {
    /// When false, depth/stencil storeOp is DontCare — the pass writes are
    /// visible during the pass, but tile contents are not flushed back on end.
    /// Safe only if the caller is certain no draw in this pass writes depth.
    bool depth_store{true};
};

class RenderManager {
    static constexpr u32 NumColorFormats = static_cast<u32>(VideoCore::PixelFormat::NumColorFormat);
    static constexpr u32 NumDepthFormats = static_cast<u32>(VideoCore::PixelFormat::NumDepthFormat);

public:
    explicit RenderManager(const Instance& instance, Scheduler& scheduler);
    ~RenderManager();

    /// Begins a new renderpass with the provided framebuffer as render target.
    void BeginRendering(const Framebuffer* framebuffer, Common::Rectangle<u32> draw_rect,
                        RenderHints hints = {});

    /// Begins a new renderpass with the provided render state.
    void BeginRendering(const RenderPass& new_pass);

    /// Exits from any currently active renderpass instance
    void EndRendering();

    /// Upgrades the in-flight renderpass from depth-DontCare to depth-Store if it
    /// was begun with depth_store=false. No-op otherwise. Must be called BEFORE
    /// any scheduler record that may write depth in the current pass.
    void EnsureDepthStore();

    /// Returns the renderpass associated with the color-depth format pair
    vk::RenderPass GetRenderpass(VideoCore::PixelFormat color, VideoCore::PixelFormat depth,
                                 bool is_clear, bool depth_store = true);

private:
    /// Creates a renderpass configured appropriately and stores it in cached_renderpasses
    vk::UniqueRenderPass CreateRenderPass(vk::Format color, vk::Format depth,
                                          vk::AttachmentLoadOp load_op, bool depth_store) const;

private:
    const Instance& instance;
    Scheduler& scheduler;
    // Axes: [color_fmt][depth_fmt][is_clear][depth_store]
    vk::UniqueRenderPass cached_renderpasses[NumColorFormats + 1][NumDepthFormats + 1][2][2];
    std::mutex cache_mutex;
    std::array<vk::Image, 2> images;
    std::array<vk::ImageAspectFlags, 2> aspects;
    bool shadow_rendering{};
    RenderPass pass{};
    u32 num_draws{};
    const Framebuffer* cached_framebuffer{nullptr};

    // Predict-and-restart state for depth storeOp. When the active pass was begun
    // with depth_store=false, remember the framebuffer so we can restart with
    // depth_store=true if a later draw requires it.
    bool pass_depth_store_predicted{true};
    const Framebuffer* pass_framebuffer{nullptr};
};

} // namespace Vulkan
