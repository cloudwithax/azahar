// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstddef>
#include "audio_core/dsp_interface.h"
#include "audio_core/sink.h"
#include "audio_core/sink_details.h"
#include "common/assert.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/dumping/backend.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace AudioCore {

DspInterface::DspInterface(Core::System& system_) : system(system_) {}

DspInterface::~DspInterface() = default;

void DspInterface::SetSink(AudioCore::SinkType sink_type, std::string_view audio_device) {
    // Dispose of the current sink first to avoid contention.
    sink.reset();

    sink = AudioCore::GetSinkDetails(sink_type).create_sink(audio_device);
    sink->SetCallback(
        [this](s16* buffer, std::size_t num_frames) { OutputCallback(buffer, num_frames); });
    time_stretcher.SetOutputSampleRate(sink->GetNativeSampleRate());
}

Sink& DspInterface::GetSink() {
    ASSERT(sink);
    return *sink.get();
}

void DspInterface::EnableStretching(bool enable) {
    enable_time_stretching = enable;
}

void DspInterface::OutputFrame(StereoFrame16 frame) {
    if (!sink) {
        return;
    }

    if (sink->ImmediateSubmission()) {
        sink->PushSamples(frame.data(), frame.size());
    } else {
        fifo.Push(frame.data(), frame.size());
    }

    auto video_dumper = system.GetVideoDumper();
    if (video_dumper && video_dumper->IsDumping()) {
        video_dumper->AddAudioFrame(std::move(frame));
    }
}

void DspInterface::OutputSample(std::array<s16, 2> sample) {
    if (!sink) {
        return;
    }

    if (sink->ImmediateSubmission()) {
        sink->PushSamples(&sample, 1);
    } else {
        fifo.Push(&sample, 1);
    }

    auto video_dumper = system.GetVideoDumper();
    if (video_dumper && video_dumper->IsDumping()) {
        video_dumper->AddAudioSample(std::move(sample));
    }
}

void DspInterface::OutputCallback(s16* buffer, std::size_t num_frames) {
    // Determine if we should stretch based on the current emulation speed.
    // Use relaxed ordering — these flags only need eventual visibility, not ordering guarantees.
    const auto perf_stats = system.GetLastPerfStats();
    const auto should_stretch =
        enable_time_stretching.load(std::memory_order_relaxed) && perf_stats.emulation_speed <= 95;
    if (performing_time_stretching.load(std::memory_order_relaxed) && !should_stretch) {
        flushing_time_stretcher.store(true, std::memory_order_relaxed);
    }
    performing_time_stretching.store(should_stretch, std::memory_order_relaxed);

    std::size_t frames_written = 0;
    if (performing_time_stretching.load(std::memory_order_relaxed)) {
        const std::size_t num_in =
            fifo.Pop(stretch_input_buffer.data(), stretch_fifo_capacity);
        frames_written = time_stretcher.Process(stretch_input_buffer.data(), num_in, buffer,
                                                num_frames);
    } else {
        if (flushing_time_stretcher.load(std::memory_order_relaxed)) {
            time_stretcher.Flush();
            frames_written = time_stretcher.Process(nullptr, 0, buffer, num_frames);
            flushing_time_stretcher.store(false, std::memory_order_relaxed);

            // Make sure any frames that did not fit are cleared from the time stretcher,
            // so that they do not bleed into the next time the stretcher is enabled.
            time_stretcher.Clear();
        }
        frames_written += fifo.Pop(buffer, num_frames - frames_written);
    }

    if (frames_written > 0) {
        std::memcpy(&last_frame[0], buffer + 2 * (frames_written - 1), 2 * sizeof(s16));
    }

    // Hold last emitted frame; this prevents popping.
    for (std::size_t i = frames_written; i < num_frames; i++) {
        std::memcpy(buffer + 2 * i, &last_frame[0], 2 * sizeof(s16));
    }

    // Implementation of the hardware volume slider
    // A cubic curve is used to approximate a linear change in human-perceived loudness
    const float linear_volume = std::clamp(Settings::Volume(), 0.0f, 1.0f);
    if (linear_volume != 1.0) {
        const float volume_scale_factor = linear_volume * linear_volume * linear_volume;
#if defined(__ARM_NEON)
        const float32x4_t vsf = vdupq_n_f32(volume_scale_factor);
        std::size_t i = 0;
        for (; i + 8 <= num_frames; i += 8) {
            int16x8_t s = vld1q_s16(buffer + i * 2);
            int16x4_t lo = vget_low_s16(s);
            int16x4_t hi = vget_high_s16(s);
            int32x4_t lo32 = vmovl_s16(lo);
            int32x4_t hi32 = vmovl_s16(hi);
            float32x4_t flo = vmulq_f32(vcvtq_f32_s32(lo32), vsf);
            float32x4_t fhi = vmulq_f32(vcvtq_f32_s32(hi32), vsf);
            int32x4_t lo_out = vcvtq_s32_f32(flo);
            int32x4_t hi_out = vcvtq_s32_f32(fhi);
            int16x4_t nlo = vqmovn_s32(lo_out);
            int16x4_t nhi = vqmovn_s32(hi_out);
            vst1q_s16(buffer + i * 2, vcombine_s16(nlo, nhi));
        }
        for (; i < num_frames; i++) {
            buffer[i * 2 + 0] = static_cast<s16>(buffer[i * 2 + 0] * volume_scale_factor);
            buffer[i * 2 + 1] = static_cast<s16>(buffer[i * 2 + 1] * volume_scale_factor);
        }
#else
        for (std::size_t i = 0; i < num_frames; i++) {
            buffer[i * 2 + 0] = static_cast<s16>(buffer[i * 2 + 0] * volume_scale_factor);
            buffer[i * 2 + 1] = static_cast<s16>(buffer[i * 2 + 1] * volume_scale_factor);
        }
#endif
    }
}

} // namespace AudioCore
