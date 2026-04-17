// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_render_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

namespace Vulkan {

// Flush threshold for tiler GPUs (Mali/Adreno/PowerVR). Too low wastes time on
// vkQueueSubmit overhead; too high risks tile buffer pressure. 30 draws balances
// submit frequency against tile utilization on Mali G52.
constexpr u32 MinDrawsToFlush = 30;

using VideoCore::PixelFormat;
using VideoCore::SurfaceType;

RenderManager::RenderManager(const Instance& instance, Scheduler& scheduler)
    : instance{instance}, scheduler{scheduler} {}

RenderManager::~RenderManager() = default;

void RenderManager::BeginRendering(const Framebuffer* framebuffer,
                                   Common::Rectangle<u32> draw_rect, RenderHints hints) {
    // Fast path: if we're already in a render pass for this exact framebuffer
    // AND the prediction is compatible (we don't need to upgrade from DontCare
    // to Store), just increment the draw count.
    if (cached_framebuffer == framebuffer && pass.render_pass &&
        (pass_depth_store_predicted || !hints.depth_store)) [[likely]] {
        num_draws++;
        return;
    }

    // Select the correct renderpass variant. The default (hints.depth_store=true)
    // matches framebuffer->RenderPass(); the opt-in DontCare variant is fetched
    // from the cache — renderpass compatibility rules (Vulkan spec §8.2) state
    // that storeOp does not affect compatibility, so the framebuffer handle and
    // pipeline objects remain valid across variants.
    vk::RenderPass selected_rp = framebuffer->RenderPass();
    if (!hints.depth_store && framebuffer->Format(VideoCore::SurfaceType::Depth) !=
                                  VideoCore::PixelFormat::Invalid) {
        const auto color_fmt = framebuffer->Format(VideoCore::SurfaceType::Color);
        const auto depth_fmt = framebuffer->Format(VideoCore::SurfaceType::Depth);
        selected_rp = GetRenderpass(color_fmt, depth_fmt, false, /*depth_store=*/false);
    }

    const vk::Rect2D render_area = {
        .offset{
            .x = 0,
            .y = 0,
        },
        .extent{
            .width = framebuffer->Width(),
            .height = framebuffer->Height(),
        },
    };
    const RenderPass new_pass = {
        .framebuffer = framebuffer->Handle(),
        .render_pass = selected_rp,
        .render_area = render_area,
        .clear = {},
        .do_clear = false,
    };
    images = framebuffer->Images();
    aspects = framebuffer->Aspects();
    shadow_rendering = framebuffer->shadow_rendering;
    BeginRendering(new_pass);

    // Set tracking state AFTER the inner call: EndRendering() inside inner
    // BeginRendering resets these fields to defaults, which would otherwise
    // clobber our predictions if we set them before.
    cached_framebuffer = framebuffer;
    pass_framebuffer = framebuffer;
    pass_depth_store_predicted = hints.depth_store;
}

void RenderManager::EnsureDepthStore() {
    // Only upgrade if we're in an active pass that was begun with depth_store=false.
    if (!pass.render_pass || pass_depth_store_predicted || !pass_framebuffer) {
        return;
    }
    // Our prediction (no depth writes) was wrong. End the DontCare pass (nothing
    // has been written to depth yet, so DontCare is still valid) and restart the
    // same framebuffer with depth_store=true. Callers MUST invoke this BEFORE
    // recording any scheduler command that writes depth.
    const Framebuffer* fb = pass_framebuffer;
    EndRendering();
    BeginRendering(fb, {}, RenderHints{.depth_store = true});
}

void RenderManager::BeginRendering(const RenderPass& new_pass) {
    if (pass == new_pass) [[likely]] {
        num_draws++;
        return;
    }

    EndRendering();
    scheduler.Record([info = new_pass](vk::CommandBuffer cmdbuf) {
        const vk::RenderPassBeginInfo renderpass_begin_info = {
            .renderPass = info.render_pass,
            .framebuffer = info.framebuffer,
            .renderArea = info.render_area,
            .clearValueCount = info.do_clear ? 1u : 0u,
            .pClearValues = &info.clear,
        };
        cmdbuf.beginRenderPass(renderpass_begin_info, vk::SubpassContents::eInline);
    });

    pass = new_pass;
}

void RenderManager::EndRendering() {
    if (!pass.render_pass) {
        return;
    }

    // The subpass dependencies declared in CreateRenderPass handle the
    // execution and memory dependencies between render passes. For non-shadow
    // rendering the driver uses these to pipeline passes without explicit
    // barriers. Shadow rendering uses storage image writes (eShaderWrite) which
    // the subpass dependencies don't cover, so we still emit an explicit barrier
    // in that case.
    scheduler.Record([images = images, aspects = aspects,
                      shadow_rendering = shadow_rendering](vk::CommandBuffer cmdbuf) {
        cmdbuf.endRenderPass();

        if (!shadow_rendering) {
            return;
        }

        // Shadow rendering writes via storage images in the fragment shader,
        // which is outside the subpass dependency's access scope. Emit an
        // explicit barrier for the shader write → shader read transition.
        u32 num_barriers = 0;
        std::array<vk::ImageMemoryBarrier, 2> barriers;
        for (u32 i = 0; i < images.size(); i++) {
            if (!images[i]) {
                continue;
            }
            if (!(aspects[i] & vk::ImageAspectFlagBits::eColor)) {
                continue;
            }
            barriers[num_barriers++] = vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = images[i],
                .subresourceRange{
                    .aspectMask = aspects[i],
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
        }
        if (num_barriers > 0) {
            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                                   vk::PipelineStageFlagBits::eFragmentShader,
                                   vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr,
                                   num_barriers, barriers.data());
        }
    });

    // Reset state.
    pass.render_pass = VK_NULL_HANDLE;
    images = {};
    aspects = {};
    shadow_rendering = false;
    cached_framebuffer = nullptr;
    pass_framebuffer = nullptr;
    pass_depth_store_predicted = true;

    // The Mali guide recommends flushing at the end of each major renderpass
    // Testing has shown this has a significant effect on rendering performance
    if (num_draws > MinDrawsToFlush && instance.ShouldFlush()) {
        scheduler.Flush();
        num_draws = 0;
    }
}

vk::RenderPass RenderManager::GetRenderpass(VideoCore::PixelFormat color,
                                            VideoCore::PixelFormat depth, bool is_clear,
                                            bool depth_store) {
    std::scoped_lock lock{cache_mutex};

    const u32 color_index =
        color == VideoCore::PixelFormat::Invalid ? NumColorFormats : static_cast<u32>(color);
    const u32 depth_index =
        depth == VideoCore::PixelFormat::Invalid
            ? NumDepthFormats
            : (static_cast<u32>(depth - VideoCore::PixelFormat::NumColorFormat));

    ASSERT_MSG(color_index <= NumColorFormats && depth_index <= NumDepthFormats,
               "Invalid color index {} and/or depth_index {}", color_index, depth_index);

    vk::UniqueRenderPass& renderpass =
        cached_renderpasses[color_index][depth_index][is_clear][depth_store];
    if (!renderpass) {
        const vk::Format color_format = instance.GetTraits(color).native;
        const vk::Format depth_format = instance.GetTraits(depth).native;
        const vk::AttachmentLoadOp load_op =
            is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
        renderpass = CreateRenderPass(color_format, depth_format, load_op, depth_store);
    }

    return *renderpass;
}

vk::UniqueRenderPass RenderManager::CreateRenderPass(vk::Format color, vk::Format depth,
                                                     vk::AttachmentLoadOp load_op,
                                                     bool depth_store) const {
    u32 attachment_count = 0;
    std::array<vk::AttachmentDescription, 2> attachments;

    bool use_color = false;
    vk::AttachmentReference color_attachment_ref{};
    bool use_depth = false;
    vk::AttachmentReference depth_attachment_ref{};

    if (color != vk::Format::eUndefined) {
        attachments[attachment_count] = vk::AttachmentDescription{
            .format = color,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            // Use eGeneral for initial/final so the rest of the codebase can
            // reference images in eGeneral without extra transitions, but use
            // eColorAttachmentOptimal during the subpass. On tile-based GPUs
            // (Mali) the driver uses the subpass layout to keep data in tile
            // memory, which is the critical bandwidth optimization.
            .initialLayout = vk::ImageLayout::eGeneral,
            .finalLayout = vk::ImageLayout::eGeneral,
        };

        color_attachment_ref = vk::AttachmentReference{
            .attachment = attachment_count++,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };

        use_color = true;
    }

    if (depth != vk::Format::eUndefined) {
        const vk::AttachmentStoreOp depth_store_op =
            depth_store ? vk::AttachmentStoreOp::eStore : vk::AttachmentStoreOp::eDontCare;
        attachments[attachment_count] = vk::AttachmentDescription{
            .format = depth,
            .loadOp = load_op,
            .storeOp = depth_store_op,
            .stencilLoadOp = load_op,
            .stencilStoreOp = depth_store_op,
            .initialLayout = vk::ImageLayout::eGeneral,
            .finalLayout = vk::ImageLayout::eGeneral,
        };

        depth_attachment_ref = vk::AttachmentReference{
            .attachment = attachment_count++,
            .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        };

        use_depth = true;
    }

    const vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = use_color ? 1u : 0u,
        .pColorAttachments = &color_attachment_ref,
        .pResolveAttachments = 0,
        .pDepthStencilAttachment = use_depth ? &depth_attachment_ref : nullptr,
    };

    // Declare explicit subpass dependencies so the driver can pipeline render
    // passes back-to-back without inserting implicit full-pipeline stalls.
    // On tile-based GPUs (Mali G52) this is critical — without these, the driver
    // must conservatively assume a full dependency between every pair of passes.
    std::array<vk::SubpassDependency, 2> dependencies;
    u32 dependency_count = 0;

    // External → subpass 0: wait for previous pass's writes to complete before
    // we start reading/writing attachments.
    dependencies[dependency_count++] = vk::SubpassDependency{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits::eEarlyFragmentTests |
                        vk::PipelineStageFlagBits::eLateFragmentTests,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits::eEarlyFragmentTests |
                        vk::PipelineStageFlagBits::eLateFragmentTests,
        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite |
                         vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead |
                         vk::AccessFlagBits::eColorAttachmentWrite |
                         vk::AccessFlagBits::eDepthStencilAttachmentRead |
                         vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
    };

    // Subpass 0 → external: make our writes visible to subsequent shader reads
    // and attachment operations.
    dependencies[dependency_count++] = vk::SubpassDependency{
        .srcSubpass = 0,
        .dstSubpass = VK_SUBPASS_EXTERNAL,
        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits::eEarlyFragmentTests |
                        vk::PipelineStageFlagBits::eLateFragmentTests,
        .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader |
                        vk::PipelineStageFlagBits::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits::eEarlyFragmentTests |
                        vk::PipelineStageFlagBits::eLateFragmentTests,
        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite |
                         vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead |
                         vk::AccessFlagBits::eColorAttachmentRead |
                         vk::AccessFlagBits::eColorAttachmentWrite |
                         vk::AccessFlagBits::eDepthStencilAttachmentRead |
                         vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
    };

    const vk::RenderPassCreateInfo renderpass_info = {
        .attachmentCount = attachment_count,
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = dependency_count,
        .pDependencies = dependencies.data(),
    };

    return instance.GetDevice().createRenderPassUnique(renderpass_info);
}

} // namespace Vulkan
