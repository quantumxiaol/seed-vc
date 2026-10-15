"""Source-guided masking must preserve singing, timing, precision and originals."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from scripts.mask_svc_silence import (
    MaskSettings, build_gain, main, quiet_intervals, validate_alignment,
)


class SilenceMaskTests(unittest.TestCase):
    def test_quiet_voice_antiphase_stereo_and_short_pause_are_preserved(self):
        rate = 8000
        time = np.arange(rate * 4) / rate
        source = np.zeros_like(time, dtype=np.float32)
        # RMS ~ -57 dBFS: quiet singing must survive a -60 dBFS gate.
        voice = (time >= 1) & (time < 3)
        source[voice] = .002 * np.sin(2 * np.pi * 200 * time[voice])
        source[(time >= 1.8) & (time < 1.9)] = 0  # Short inter-syllable pause.
        stereo = np.column_stack((source, -source))  # A mono mix would cancel.
        settings = MaskSettings()
        quiet, _, _ = quiet_intervals(stereo, rate, settings)
        gain, segments = build_gain(quiet, 4, len(source), rate, settings)
        self.assertEqual(len(segments), 2)
        np.testing.assert_array_equal(gain[int(.9 * rate):int(3.2 * rate)], 1)
        np.testing.assert_array_equal(gain[:int(.8 * rate)], 0)
        np.testing.assert_array_equal(gain[int(3.3 * rate):], 0)

    def test_internal_gap_has_protected_edges_and_smooth_fades(self):
        rate = 1000
        gain, segments = build_gain([(1, 3)], 4, 4000, rate, MaskSettings())
        np.testing.assert_array_equal(gain[:1200], 1)
        np.testing.assert_array_equal(gain[1230:2870], 0)
        np.testing.assert_array_equal(gain[2900:], 1)
        self.assertTrue(np.all(np.diff(gain[1199:1231]) <= 0))
        self.assertTrue(np.all(np.diff(gain[2869:2901]) >= 0))
        self.assertLess(np.max(np.abs(np.diff(gain))), .06)
        self.assertEqual(len(segments), 1)

    def test_gap_without_room_for_protection_is_untouched(self):
        gain, segments = build_gain([(1, 1.3)], 2, 2000, 1000, MaskSettings())
        np.testing.assert_array_equal(gain, 1)
        self.assertEqual(segments, [])

    def test_silent_and_continuous_sources_including_short_clips(self):
        for duration, level, expected in [(2., 0., 0), (2., .01, 1), (.01, 0., 1)]:
            with self.subTest(duration=duration, level=level):
                source = np.full(round(8000 * duration), level, dtype=np.float32)
                intervals, _, _ = quiet_intervals(source, 8000, MaskSettings())
                gain, _ = build_gain(intervals, duration, len(source), 8000, MaskSettings())
                np.testing.assert_array_equal(gain, expected)

    def test_alignment_accepts_frame_rounding_but_rejects_changed_duration(self):
        validate_alignment(48000 * 4, 48000, 44100 * 4 - 512, 44100)
        with self.assertRaisesRegex(ValueError, "时长不匹配"):
            validate_alignment(48000 * 4, 48000, 44100 * 3, 44100)

    def test_cli_preserves_samples_format_and_originals_at_different_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, converted_path = root / "source.wav", root / "converted.wav"
            source = np.zeros(48000 * 4, dtype=np.float32)
            source[48000:48000 * 3] = .1
            sf.write(source_path, source, 48000, subtype="FLOAT")
            rng = np.random.default_rng(0)
            for subtype, dtype in [("FLOAT", "float32"), ("DOUBLE", "float64")]:
                with self.subTest(subtype=subtype):
                    converted = rng.uniform(-.5, .5, (44100 * 4 - 512, 2)).astype(dtype)
                    sf.write(converted_path, converted, 44100, subtype=subtype)
                    original_bytes = converted_path.read_bytes()
                    output = root / f"masked_{subtype}.wav"
                    args = ["--source", str(source_path), "--converted", str(converted_path),
                            "--output", str(output)]
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(main(args), 0)
                    result, rate = sf.read(output, always_2d=True, dtype=dtype)
                    self.assertEqual(rate, 44100)
                    self.assertEqual(sf.info(output).subtype, subtype)
                    self.assertEqual(result.shape, converted.shape)
                    np.testing.assert_array_equal(result[44100:44100 * 3], converted[44100:44100 * 3])
                    np.testing.assert_array_equal(result[:44100 // 2], 0)
                    np.testing.assert_array_equal(result[-44100 // 2:], 0)
                    self.assertEqual(converted_path.read_bytes(), original_bytes)
                    report = json.loads(output.with_suffix(".mask.json").read_text())
                    self.assertEqual(len(report["intervals"]), 2)
                    saved = output.read_bytes()
                    with redirect_stderr(io.StringIO()):
                        self.assertEqual(main(args), 1)  # Never overwrite an earlier export.
                    self.assertEqual(output.read_bytes(), saved)
                    with redirect_stderr(io.StringIO()):
                        self.assertEqual(main(args[:-1] + [str(source_path)]), 1)

    def test_invalid_settings_are_rejected(self):
        for settings in [MaskSettings(threshold_db=float("nan")), MaskSettings(fade_ms=-1),
                         MaskSettings(min_silence_ms=0), MaskSettings(threshold_db=5)]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                settings.validate()


if __name__ == "__main__":
    unittest.main()
