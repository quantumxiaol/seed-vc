"""Offline checks for the Hub 1.x / Transformers 5 model-loading contracts.

Run: uv run --env-file .env python -m unittest discover -s tests -v
Tiny random checkpoints test API compatibility, not conversion quality.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WhisperConfig, WhisperFeatureExtractor, WhisperModel

from modules.bigvgan.bigvgan import BigVGAN
from modules.bigvgan.env import AttrDict


class ModelCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        torch.manual_seed(42)
        original_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.addClassCleanup(torch.set_num_threads, original_threads)

        cls.vocoder = BigVGAN(AttrDict({
            "resblock": "1", "num_mels": 4, "upsample_initial_channel": 32,
            "upsample_rates": [2, 2], "upsample_kernel_sizes": [4, 4],
            "resblock_kernel_sizes": [3], "resblock_dilation_sizes": [[1, 3, 5]],
            "activation": "snakebeta", "snake_logscale": True,
            "use_tanh_at_final": False, "use_bias_at_final": False,
        })).eval()
        cls.vocoder.save_pretrained(cls.root / "bigvgan")
        cls.mel = torch.randn(1, 4, 16)
        with torch.no_grad():
            cls.expected_audio = cls.vocoder(cls.mel)

        cls.whisper = WhisperModel(WhisperConfig(
            vocab_size=32, pad_token_id=0, bos_token_id=1, eos_token_id=2,
            decoder_start_token_id=1, d_model=16, encoder_layers=1, decoder_layers=1,
            encoder_attention_heads=2, decoder_attention_heads=2,
            encoder_ffn_dim=32, decoder_ffn_dim=32, max_source_positions=50,
        )).eval()
        cls.whisper.save_pretrained(cls.root / "whisper")
        WhisperFeatureExtractor(chunk_length=1).save_pretrained(cls.root / "whisper")

    def check_vocoder(self, model):
        with torch.no_grad():
            audio = model.eval()(self.mel)
        self.assertEqual(audio.shape, (1, 1, 64))
        torch.testing.assert_close(audio, self.expected_audio)

    def test_bigvgan_local_checkpoint(self):
        self.check_vocoder(BigVGAN.from_pretrained(
            self.root / "bigvgan", local_files_only=True, use_cuda_kernel=False,
        ))

    def test_bigvgan_hub_download_api(self):
        # Exercise the actual Hub mixin and validate calls against the installed
        # hf_hub_download signature, while serving fixture files without network.
        def cached_file(repo_id, filename, **kwargs):
            return str(self.root / "bigvgan" / filename)

        with patch("huggingface_hub.hub_mixin.hf_hub_download", autospec=True,
                   side_effect=cached_file), patch(
            "modules.bigvgan.bigvgan.hf_hub_download", autospec=True, side_effect=cached_file,
        ):
            model = BigVGAN.from_pretrained("test/tiny-vocoder", local_files_only=True)
        self.check_vocoder(model)

    def check_whisper(self, device, dtype):
        extractor = AutoFeatureExtractor.from_pretrained(self.root / "whisper", local_files_only=True)
        wave = np.sin(2 * np.pi * 220 * np.arange(8000) / 16000).astype(np.float32) * 0.1
        features = extractor([wave], sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        model = WhisperModel.from_pretrained(
            self.root / "whisper", local_files_only=True, dtype=dtype,
        ).eval().to(device)
        del model.decoder  # Same encoder-only usage as SVC.
        inputs = model._mask_input_features(features.input_features, attention_mask=features.attention_mask)
        with torch.no_grad():
            expected = self.whisper.encoder(features.input_features, return_dict=True).last_hidden_state
            actual = model.encoder(inputs.to(device=device, dtype=dtype),
                                   output_attentions=False, output_hidden_states=False,
                                   return_dict=True).last_hidden_state
        self.assertEqual(actual.shape, (1, 50, 16))
        self.assertTrue(torch.isfinite(actual).all().item())
        tolerance = 0.01 if dtype == torch.float16 else 1e-5
        torch.testing.assert_close(actual.cpu().float(), expected, rtol=tolerance, atol=tolerance)

    def test_whisper_cpu(self):
        self.check_whisper("cpu", torch.float32)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is not available")
    def test_whisper_mps_fp16(self):
        self.check_whisper("mps", torch.float16)


if __name__ == "__main__":
    unittest.main()
