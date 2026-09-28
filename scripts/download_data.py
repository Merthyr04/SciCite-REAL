#!/usr/bin/env python3
"""Download the SciCite / ACL-ARC datasets from the official openAI S3 bucket.

Uses the same public URLs documented in the original allenai/SciCite repo.
SciCite is the primary dataset; ACL-ARC is used for cross-dataset evaluation.

Usage:
    python scripts/download_data.py            # download SciCite (default)
    python scripts/download_data.py --include-acl-arc
    python scripts/download_data.py --data-dir data/raw
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

SCICITE_URL = (
    "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz"
)
ACLARC_URL = (
    "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/acl-arc.tar.gz"
)


def download_and_extract(url: str, dest_dir: Path, name: str) -> None:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tarball = dest_dir / f"{name}.tar.gz"
    if not tarball.exists():
        print(f"[download] {url}")
        urllib.request.urlretrieve(url, str(tarball))
    else:
        print(f"[skip] {tarball} already present")

    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(dest_dir)
    print(f"[extract] {name} -> {dest_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download citation-intent datasets")
    ap.add_argument("--include-acl-arc", action="store_true",
                    help="Also download ACL-ARC")
    ap.add_argument("--data-dir", default="data/raw",
                    help="Root where tarballs are extracted")
    args = ap.parse_args()

    download_and_extract(SCICITE_URL, Path(args.data_dir) / "scicite", "scicite")
    if args.include_acl_arc:
        download_and_extract(ACLARC_URL, Path(args.data_dir) / "acl-arc", "acl-arc")


if __name__ == "__main__":
    main()