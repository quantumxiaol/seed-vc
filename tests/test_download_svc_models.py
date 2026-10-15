"""Exercise resume and HF cache compatibility using an in-memory HTTP transport."""

from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from huggingface_hub import HfFileMetadata, hf_hub_download

from scripts import download_svc_models as downloader


class InterruptedStream(httpx.SyncByteStream):
    def __iter__(self):
        yield b"abcd"
        raise KeyboardInterrupt


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name).resolve()
        self.payload = b"abcdefghijkl"
        self.etag = hashlib.sha256(self.payload).hexdigest()
        self.metadata = HfFileMetadata(
            commit_hash="a" * 40, etag=self.etag,
            location="https://fixture.invalid/model", size=len(self.payload), xet_file_data=None,
        )
        self.partial = self.cache / "models--test--svc" / "blobs" / (self.etag + ".incomplete")
        self.start_patch("scripts.download_svc_models.get_hf_file_metadata", return_value=self.metadata)
        # Small chunks let tiny fixtures exercise streaming interruption and resume.
        self.start_patch("huggingface_hub.constants.DOWNLOAD_CHUNK_SIZE", 4)
        self.start_patch("huggingface_hub.constants.HF_HUB_DISABLE_TELEMETRY", True)
        self.output = io.StringIO()
        capture = redirect_stdout(self.output)
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

    def start_patch(self, name, *args, **kwargs):
        patcher = patch(name, *args, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def serve(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        self.start_patch("huggingface_hub.utils._http.get_session", return_value=client)

    def fetch(self):
        return downloader.download_file("test/svc", "model.pth", self.cache)

    def assert_cached(self, path):
        self.assertEqual(Path(path).read_bytes(), self.payload)
        # Verify the real HF reader can resolve main and the snapshot offline.
        self.assertEqual(hf_hub_download(
            "test/svc", "model.pth", cache_dir=self.cache, local_files_only=True,
        ), path)
        self.assertFalse(self.partial.exists())
        with patch.object(downloader, "get_hf_file_metadata", side_effect=AssertionError("network")):
            self.assertEqual(self.fetch(), path)

    def test_interrupt_then_resume_into_standard_cache(self):
        ranges = []

        def handler(request):
            ranges.append(request.headers.get("Range"))
            if len(ranges) == 1:
                return httpx.Response(200, headers={"Content-Length": "12"}, stream=InterruptedStream())
            self.assertEqual(request.headers["Range"], "bytes=4-")
            return httpx.Response(206, headers={"Content-Range": "bytes 4-11/12"}, content=self.payload[4:])

        self.serve(handler)
        with self.assertRaises(KeyboardInterrupt):
            self.fetch()
        self.assertEqual(self.partial.read_bytes(), b"abcd")
        self.assert_cached(self.fetch())
        self.assertEqual(ranges, [None, "bytes=4-"])

    def test_server_ignoring_range_restarts_without_duplicate_bytes(self):
        self.partial.parent.mkdir(parents=True)
        self.partial.write_bytes(self.payload[:4])

        def handler(request):
            self.assertEqual(request.headers["Range"], "bytes=4-")
            return httpx.Response(200, content=self.payload)

        self.serve(handler)
        self.assert_cached(self.fetch())

    def test_short_response_retries_from_saved_bytes(self):
        ranges = []

        def handler(request):
            ranges.append(request.headers.get("Range"))
            if len(ranges) == 1:
                return httpx.Response(200, headers={"Content-Length": "12"}, content=self.payload[:4])
            return httpx.Response(206, headers={"Content-Range": "bytes 4-11/12"}, content=self.payload[4:])

        self.serve(handler)
        with patch.object(downloader.time, "sleep"):
            path = downloader.download_with_retries("test/svc", "model.pth", self.cache, 2)
        self.assert_cached(path)
        self.assertEqual(ranges, [None, "bytes=4-"])

    def test_corrupt_content_never_becomes_cached_weight(self):
        self.serve(lambda request: httpx.Response(200, content=b"x" * len(self.payload)))
        with self.assertRaisesRegex(ValueError, "校验失败"):
            self.fetch()
        self.assertTrue(self.partial.exists())
        self.assertFalse((self.cache / "models--test--svc" / "refs" / "main").exists())

    def test_git_blob_config_hash(self):
        config = self.cache / "config.json"
        config.write_bytes(b"{}\n")
        etag = hashlib.sha1(b"blob 3\0{}\n").hexdigest()
        downloader.verify_file(config, etag, 3)

    def test_check_detects_manual_weight_and_lists_missing_aux_without_network(self):
        (self.cache / downloader.MAIN_WEIGHT).write_bytes(b"manual checkpoint")
        with patch.object(downloader.constants, "HF_HOME", str(self.cache)), patch.object(
            downloader.constants, "HF_HUB_CACHE", str(self.cache / "hub"),
        ), patch.object(downloader, "get_hf_file_metadata", side_effect=AssertionError("network")):
            self.assertEqual(downloader.main(["--check"]), 1)
        output = self.output.getvalue()
        self.assertIn("跳过下载", output)
        self.assertIn("缺少 7 个文件", output)
        self.assertNotIn("Plachta/Seed-VC", output)


if __name__ == "__main__":
    unittest.main()
