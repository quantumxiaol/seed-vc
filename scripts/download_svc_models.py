"""Download the 44.1 kHz SVC weights with progress and persistent HTTP resume.

Run from the project root:
    uv run --env-file .env scripts/download_svc_models.py
"""

import argparse
import hashlib
import os
from pathlib import Path
import re
import shlex
import sys
import time
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "modelsweights"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

import httpx
from filelock import FileLock
from huggingface_hub import (
    constants, get_hf_file_metadata, hf_hub_url, try_to_load_from_cache,
)
from huggingface_hub.file_download import http_get

# Hub buffers 10 MiB per update by default, which looks stalled on slow links.
# This downloader runs in its own process; use smaller chunks for progress/resume.
constants.DOWNLOAD_CHUNK_SIZE = 64 * 1024


MAIN_WEIGHT = "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
MAIN_CONFIG = "configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
SVC_FILES = {
    "Plachta/Seed-VC": [MAIN_WEIGHT, Path(MAIN_CONFIG).name],
    "funasr/campplus": ["campplus_cn_common.bin"],
    "lj1995/VoiceConversionWebUI": ["rmvpe.pt"],
    "nvidia/bigvgan_v2_44khz_128band_512x": ["config.json", "bigvgan_generator.pt"],
    "openai/whisper-small": ["config.json", "preprocessor_config.json", "model.safetensors"],
}


def verify_file(path, etag, size):
    """Hub ETags are SHA-256 for LFS weights or Git blob SHA-1 for small files."""
    if path.stat().st_size != size:
        raise OSError(f"文件尚未完整下载：{path.stat().st_size}/{size} bytes")
    digest = hashlib.sha256() if len(etag) == 64 else hashlib.sha1()
    if len(etag) == 40:
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != etag:
        raise ValueError(f"文件校验失败，请删除此不完整文件后重试：{path}")


def publish_snapshot(repo_dir, blob, commit, filename):
    """Publish the standard HF cache layout, so all model loaders can reuse it."""
    pointer = repo_dir / "snapshots" / commit / filename
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_name(f".{pointer.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.symlink_to(os.path.relpath(blob, pointer.parent))
        temporary.replace(pointer)
    finally:
        temporary.unlink(missing_ok=True)
    ref = repo_dir / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    temporary = ref.with_name(f".main.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(commit)
        temporary.replace(ref)
    finally:
        temporary.unlink(missing_ok=True)
    return str(pointer)


def download_file(repo_id, filename, cache_dir):
    cached = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir)
    if isinstance(cached, str):
        return cached
    metadata = get_hf_file_metadata(hf_hub_url(repo_id, filename), token=False)
    if not (
        re.fullmatch(r"[0-9a-f]{40}", metadata.commit_hash or "")
        and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", metadata.etag or "")
        and metadata.size is not None
    ):
        raise ValueError("服务器缺少有效的 commit / ETag / 文件大小，无法安全续传。")

    folder = "models--" + repo_id.replace("/", "--")
    repo_dir = cache_dir / folder
    blob = repo_dir / "blobs" / metadata.etag
    partial = blob.with_name(blob.name + ".incomplete")
    lock = cache_dir / ".locks" / folder / (metadata.etag + ".lock")
    blob.parent.mkdir(parents=True, exist_ok=True)
    lock.parent.mkdir(parents=True, exist_ok=True)
    print("  等待文件锁 / 准备下载…", flush=True)
    with FileLock(lock):
        if not blob.exists():
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > metadata.size:
                raise ValueError(f"临时文件超过预期大小，请删除后重试：{partial}")
            if offset:
                print(f"  从 {offset / 1024**2:.1f} MiB 处续传", flush=True)
            # Hub 1.33's high-level downloader deletes partial files on failure.
            # Keep a stable, ETag-keyed partial under a file lock instead. Its HTTP
            # helper supplies Range requests, streaming retries and a byte progress bar.
            with partial.open("ab") as stream:
                http_get(
                    metadata.location, stream, resume_size=offset,
                    expected_size=metadata.size, displayed_filename=filename,
                    headers={"Accept-Encoding": "identity"},
                )
            print("  校验文件…", flush=True)
            verify_file(partial, metadata.etag, metadata.size)
            partial.replace(blob)
        return publish_snapshot(repo_dir, blob, metadata.commit_hash, filename)


def download_with_retries(repo_id, filename, cache_dir, attempts):
    for attempt in range(1, attempts + 1):
        try:
            return download_file(repo_id, filename, cache_dir)
        except (httpx.HTTPError, OSError) as error:
            if isinstance(error, httpx.HTTPStatusError):
                status = error.response.status_code
                if status < 500 and status not in (408, 429):
                    raise
            if attempt == attempts:
                raise
            delay = min(2 ** attempt, 30)
            print(f"  下载失败（{attempt}/{attempts}）：{error}\n"
                  f"  {delay} 秒后重试，已下载部分保留。", flush=True)
            time.sleep(delay)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="仅检查本地文件，不联网")
    parser.add_argument("--only-aux", action="store_true", help="只下载四组辅助模型")
    parser.add_argument("--checkpoint", type=Path, help="已手动下载的 SVC 主权重路径")
    parser.add_argument("--attempts", type=int, default=5, help="每个文件最多尝试次数（默认 5）")
    args = parser.parse_args(argv)
    if args.attempts < 1:
        parser.error("--attempts 必须至少为 1")
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidate = Path(constants.HF_HOME) / MAIN_WEIGHT
        if candidate.is_file():
            checkpoint = candidate
    if checkpoint is not None and (not checkpoint.is_file() or checkpoint.stat().st_size == 0):
        parser.error(f"主权重不存在或为空：{checkpoint}")
    cache_dir = Path(constants.HF_HUB_CACHE).expanduser().resolve()
    print(f"Hugging Face 缓存：{cache_dir}", flush=True)
    if checkpoint is not None:
        print(f"使用手动下载的主权重，跳过下载：{checkpoint}", flush=True)
        print("启动时请指定本地权重：\n"
              "uv run --env-file .env app_svc.py "
              f"--checkpoint {shlex.quote(str(checkpoint))} "
              f"--config {MAIN_CONFIG} --fp16 False\n", flush=True)
    files = [
        (repo, filename)
        for repo, filenames in SVC_FILES.items()
        if not (repo == "Plachta/Seed-VC" and (args.only_aux or checkpoint is not None))
        for filename in filenames
    ]
    missing = []
    try:
        for index, (repo, filename) in enumerate(files, 1):
            print(f"[{index}/{len(files)}] {repo}/{filename}", flush=True)
            cached = try_to_load_from_cache(repo, filename, cache_dir=cache_dir)
            if isinstance(cached, str):
                print(f"  已缓存：{cached}", flush=True)
            elif args.check:
                missing.append(f"{repo}/{filename}")
                print("  缺失", flush=True)
            else:
                path = download_with_retries(repo, filename, cache_dir, args.attempts)
                print(f"  完成：{path}", flush=True)
    except KeyboardInterrupt:
        print("\n下载已中断，未完成部分已保留。重新执行同一命令即可续传。", file=sys.stderr)
        return 130
    except (httpx.HTTPError, OSError, ValueError) as error:
        print(f"\n下载失败：{error}\n已完成文件及未完成部分均已保留。"
              "处理上述错误后，重新执行同一命令。", file=sys.stderr)
        if "UNEXPECTED_EOF_WHILE_READING" in str(error):
            print("SSL 连接被提前关闭；镜像可能跳转回官方站点。请确认 .env 中的 "
                  "HTTP_PROXY / HTTPS_PROXY 已指向正在运行的代理，且 NO_PROXY 未排除 HF 域名。",
                  file=sys.stderr)
        return 1
    if missing:
        print(f"\n缺少 {len(missing)} 个文件。去掉 --check 即可下载。", flush=True)
        return 1
    print("\n所选 SVC 模型文件已就绪。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
