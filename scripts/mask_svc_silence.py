"""Mute SVC artifacts in gaps identified from the aligned source dry vocal.

No model, ASR, resampling, normalization, or inference is needed. The source
and converted file must have the same timeline (inference --length-adjust 1.0).
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class MaskSettings:
    threshold_db: float = -60.0
    min_silence_ms: float = 300.0
    pre_pad_ms: float = 100.0
    post_pad_ms: float = 200.0
    fade_ms: float = 30.0

    def validate(self):
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("参数必须是有限数值。")
        if not -120 <= self.threshold_db < 0:
            raise ValueError("--threshold-db 必须在 -120（含）到 0（不含）dBFS 之间。")
        if self.min_silence_ms <= 0:
            raise ValueError("--min-silence-ms 必须大于 0。")
        if min(self.pre_pad_ms, self.post_pad_ms, self.fade_ms) < 0:
            raise ValueError("保护时间和淡入淡出时间不能为负数。")


def rms_envelope(audio, sample_rate):
    """20 ms RMS windows every 10 ms; use the loudest channel, never a mixdown."""
    if audio.ndim == 1:
        audio = audio[:, None]
    hop = max(1, round(sample_rate * .010))
    window = max(hop, round(sample_rate * .020))
    starts = np.arange(0, len(audio), hop)
    ends = np.minimum(starts + window, len(audio))
    energy = np.zeros(len(starts), dtype=np.float64)
    for channel in audio.T:
        # Float64 accumulation preserves quiet tails next to much louder singing.
        integral = np.empty(len(audio) + 1, dtype=np.float64)
        integral[0] = 0
        np.cumsum(np.square(channel, dtype=np.float64), out=integral[1:])
        energy = np.maximum(energy, (integral[ends] - integral[starts]) / (ends - starts))
    return starts / sample_rate, np.sqrt(energy), hop, window


def quiet_intervals(audio, sample_rate, settings):
    """Return sustained low-energy intervals, excluding all active window spans."""
    settings.validate()
    times, rms, hop, window = rms_envelope(audio, sample_rate)
    quiet = rms < 10 ** (settings.threshold_db / 20)
    edges = np.diff(np.r_[False, quiet, False].astype(np.int8))
    intervals = []
    for first, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        # Adjacent active windows overlap the first quiet window: protect that overlap.
        start = 0 if first == 0 else (first * hop + window - hop) / sample_rate
        end = len(audio) / sample_rate if stop == len(rms) else stop * hop / sample_rate
        if end - start >= settings.min_silence_ms / 1000:
            intervals.append((start, end))
    return intervals, times, rms


def build_gain(intervals, source_duration, output_frames, output_rate, settings):
    """Protect both vocal edges; place cosine fades entirely within the quiet gap."""
    settings.validate()
    gain = np.ones(output_frames, dtype=np.float32)
    fades = max(2, round(settings.fade_ms * output_rate / 1000)) if settings.fade_ms else 0
    ramp = (.5 - .5 * np.cos(np.linspace(0, np.pi, fades))).astype(np.float32)
    applied = []
    for start, end in intervals:
        leading = start == 0
        trailing = end >= source_duration
        first = 0 if leading else math.ceil((start + settings.post_pad_ms / 1000) * output_rate)
        stop = output_frames if trailing else math.floor((end - settings.pre_pad_ms / 1000) * output_rate)
        first, stop = max(0, min(first, output_frames)), max(0, min(stop, output_frames))
        fade_out = 0 if leading else fades
        fade_in = 0 if trailing else fades
        mute_first, mute_stop = first + fade_out, stop - fade_in
        # Leave short pauses alone when protection and fades consume the whole gap.
        if mute_first >= mute_stop:
            continue
        if fade_out:
            gain[first:mute_first] = 1 - ramp
        gain[mute_first:mute_stop] = 0
        if fade_in:
            gain[mute_stop:stop] = ramp
        applied.append({
            "source_quiet_start_s": start,
            "source_quiet_end_s": end,
            "fade_out_start_s": first / output_rate,
            "mute_start_s": mute_first / output_rate,
            "mute_end_s": mute_stop / output_rate,
            "fade_in_end_s": stop / output_rate,
        })
    return gain, applied


def validate_alignment(source_frames, source_rate, converted_frames, converted_rate):
    source_duration = source_frames / source_rate
    converted_duration = converted_frames / converted_rate
    # Seed-VC mel framing trims a few ms; never stretch a mask to hide a mismatch.
    if abs(source_duration - converted_duration) > .1:
        raise ValueError(
            f"源音频 {source_duration:.3f}s 与转换结果 {converted_duration:.3f}s 时长不匹配。"
            "请使用同一段音频、相同起点及 --length-adjust 1.0 的结果；不会自动拉伸对齐。"
        )


def read_audio(path):
    info = sf.info(path)
    # Preserve PCM32 / DOUBLE input precision too; standard SVC output is FLOAT.
    dtype = "float64" if info.subtype in {"DOUBLE", "PCM_32"} else "float32"
    audio, rate = sf.read(path, dtype=dtype, always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"音频为空或含有 NaN / Inf：{path}")
    return audio, rate


def to_db(rms):
    return 20 * np.log10(np.maximum(rms, 1e-6))


def plot_mask(path, source_times, source_rms, before, after, rate, gain, settings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before_times, before_rms, hop, _ = rms_envelope(before, rate)
    after_times, after_rms, _, _ = rms_envelope(after, rate)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})
    axes[0].plot(source_times, to_db(source_rms), color="#3566a6", lw=.8)
    axes[0].axhline(settings.threshold_db, color="#b55c22", ls="--", lw=1,
                    label=f"Silence threshold: {settings.threshold_db:g} dBFS")
    axes[0].set_title("Original dry vocal (loudest channel)", loc="left")
    axes[0].legend(loc="upper right")
    axes[1].plot(before_times, to_db(before_rms), color="#bb6c3a", lw=.8, label="Before")
    axes[1].plot(after_times, to_db(after_rms), color="#27866b", lw=.8, label="After mask")
    axes[1].set_title("Converted vocal: before / after", loc="left")
    axes[1].legend(loc="upper right")
    for ax in axes[:2]:
        ax.set_ylim(-122, 3)
        ax.set_ylabel("RMS (dBFS)")
    times = np.r_[before_times, len(gain) / rate]
    sampled_gain = np.r_[gain[::hop], gain[-1]]
    axes[2].fill_between(times, sampled_gain, color="#27866b", alpha=.25)
    axes[2].plot(times, sampled_gain, color="#27866b", lw=1)
    axes[2].set_ylabel("Gain")
    axes[2].set_yticks([0, 1], ["0 = mute", "1 = keep"])
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylim(-.05, 1.1)
    axes[2].set_xlim(0, len(gain) / rate)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="原始干声音频（不是角色参考音频）")
    parser.add_argument("--converted", type=Path, required=True, help="与原干声时间对齐的转换结果")
    parser.add_argument("--output", type=Path, help="输出 WAV；默认在转换结果旁加 _masked 后缀")
    parser.add_argument("--threshold-db", type=float, default=-60, help="静音 RMS 阈值 dBFS，默认 -60；越负越保守")
    parser.add_argument("--min-silence-ms", type=float, default=300, help="候选空白最短时长，默认 300 ms")
    parser.add_argument("--pre-pad-ms", type=float, default=100, help="句首前完整保留时长，默认 100 ms")
    parser.add_argument("--post-pad-ms", type=float, default=200, help="句尾后完整保留时长，默认 200 ms")
    parser.add_argument("--fade-ms", type=float, default=30, help="空白内部的淡入淡出时长，默认 30 ms")
    parser.add_argument("--plot", action="store_true", help="另存 RMS 电平和 mask 图（需要 matplotlib）")
    args = parser.parse_args(argv)
    settings = MaskSettings(args.threshold_db, args.min_silence_ms, args.pre_pad_ms,
                            args.post_pad_ms, args.fade_ms)
    try:
        settings.validate()
        output = args.output or args.converted.with_name(args.converted.stem + "_masked.wav")
        if output.suffix.lower() != ".wav":
            raise ValueError("--output 必须是 .wav 文件。")
        report_path = output.with_suffix(".mask.json")
        plot_path = output.with_suffix(".mask.png")
        destinations = [output, report_path] + ([plot_path] if args.plot else [])
        inputs = {args.source.resolve(), args.converted.resolve()}
        for path in destinations:
            if path.resolve() in inputs or path.exists() or path.is_symlink():
                raise ValueError(f"拒绝覆盖已有文件：{path}。请用 --output 指定新文件名。")
        source, source_rate = read_audio(args.source)
        converted, rate = read_audio(args.converted)
        validate_alignment(len(source), source_rate, len(converted), rate)
        intervals, times, rms = quiet_intervals(source, source_rate, settings)
        gain, applied = build_gain(intervals, len(source) / source_rate, len(converted), rate, settings)
        result = converted * gain[:, None]
        subtype = "DOUBLE" if converted.dtype == np.float64 else "FLOAT"
        report = {
            "source": str(args.source), "converted": str(args.converted), "output": str(output),
            "settings": asdict(settings), "source_duration_s": len(source) / source_rate,
            "output_duration_s": len(converted) / rate, "sample_rate": rate,
            "frames": len(converted), "channels": converted.shape[1], "subtype": subtype,
            "source_peak_rms_dbfs": float(to_db(rms.max())),
            "fully_muted_s": int(np.count_nonzero(gain == 0)) / rate,
            "attenuated_s_including_fades": int(np.count_nonzero(gain < 1)) / rate,
            "intervals": applied,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.plot:
            plot_mask(plot_path, times, rms, converted, result, rate, gain, settings)
        with output.open("xb") as stream:
            sf.write(stream, result, rate, format="WAV", subtype=subtype)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"输出：{output}")
        print(f"处理 {len(applied)} 段空白；完全静音 {report['fully_muted_s']:.2f} 秒。")
        print(f"保持 {rate} Hz / {converted.shape[1]} 声道 / {len(converted)} 采样帧；未归一化或重采样。")
        for interval in applied:
            print(f"  静音 {interval['mute_start_s']:.3f}–{interval['mute_end_s']:.3f}s（边界另有淡入淡出）")
        if rms.max() < 10 ** (settings.threshold_db / 20):
            print("提示：整段源音频均低于阈值。若原音频有歌声，请降低 --threshold-db 后重新处理。")
        print(f"报告：{report_path}")
        if args.plot:
            print(f"电平与 mask 图：{plot_path}")
        return 0
    except (OSError, ValueError, sf.LibsndfileError, ImportError) as exc:
        print(f"处理失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
