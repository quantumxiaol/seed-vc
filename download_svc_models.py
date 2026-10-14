"""Download only the weights used by the default 44.1 kHz SVC model.

Run from the project root: uv run --env-file .env download_svc_models.py
"""

import argparse

from huggingface_hub import constants, hf_hub_download


SVC_FILES = {
    "Plachta/Seed-VC": [
        "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
        "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
    ],
    "funasr/campplus": ["campplus_cn_common.bin"],
    "lj1995/VoiceConversionWebUI": ["rmvpe.pt"],
    "nvidia/bigvgan_v2_44khz_128band_512x": [
        "config.json",
        "bigvgan_generator.pt",
    ],
    "openai/whisper-small": [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Check the local cache without downloading"
    )
    args = parser.parse_args()
    print(f"Hugging Face cache: {constants.HF_HUB_CACHE}", flush=True)
    for repo_id, filenames in SVC_FILES.items():
        for filename in filenames:
            print(f"{repo_id}/{filename}", flush=True)
            path = hf_hub_download(
                repo_id=repo_id, filename=filename, local_files_only=args.check
            )
            print(f"  {path}", flush=True)
    print("All SVC model files are available.", flush=True)


if __name__ == "__main__":
    main()
