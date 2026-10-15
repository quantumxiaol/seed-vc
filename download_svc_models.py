"""Compatibility entry point; the downloader now lives in scripts/."""

from scripts.download_svc_models import main


if __name__ == "__main__":
    raise SystemExit(main())
