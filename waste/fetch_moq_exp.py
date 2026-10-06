#!/usr/bin/env python3
"""Fetch files from w-ahmad/MoQ-EXP HF dataset repo, skipping anything > 1MB."""

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from huggingface_hub import HfApi, list_repo_files, hf_hub_download

REPO_ID = "w-ahmad/MoQ-EXP"
REPO_TYPE = "dataset"
TOKEN = "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"
MAX_SIZE_BYTES = 1 * 1024 * 1024  # 1MB
OUT_DIR = Path("/mnt/d/Workspace/01 Projects/Mixture-of-Quants/MoQ-EXP")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
MAX_WORKERS = 8


def check_size(path: str) -> tuple[str, int]:
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{path}"
    try:
        resp = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=20)
        size = int(resp.headers.get("Content-Length", "0"))
        return path, size
    except Exception as e:
        return path, -1


def download_file(path: str, size: int) -> tuple[str, bool, str]:
    dest = OUT_DIR / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return path, True, "exists"

    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=path,
            repo_type=REPO_TYPE,
            token=TOKEN,
            local_dir=str(OUT_DIR),
            force_download=False,
        )
        if Path(downloaded_path).exists():
            return path, True, "ok"
        return path, False, f"missing_after_download:{downloaded_path}"
    except Exception as e:
        return path, False, str(e)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    api = HfApi(token=TOKEN)
    files = list(list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE, token=TOKEN))
    print(f"Found {len(files)} entries in repo")

    # Phase 1: check sizes concurrently
    sizes = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(check_size, p): p for p in files}
        for future in as_completed(futures):
            path, size = future.result()
            sizes[path] = size

    to_download = []
    skipped = 0
    skip_reasons = {}

    for path in files:
        size = sizes.get(path, -1)
        if size < 0:
            reason = "inspect_failed"
        elif size == 0:
            reason = "size_unknown"
        elif size > MAX_SIZE_BYTES:
            reason = f"too_large:{size / 1024 / 1024:.2f}MB"
        else:
            to_download.append((path, size))
            continue
        skipped += 1
        skip_reasons[path] = reason
        print(f"[SKIP] {path} ({reason})")

    print(f"\nWill download {len(to_download)} files, skip {skipped}")

    # Phase 2: download concurrently
    downloaded = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, p, s): p for p, s in to_download}
        for future in as_completed(futures):
            path, ok, msg = future.result()
            if ok:
                if msg == "exists":
                    print(f"[EXISTS] {path}")
                else:
                    print(f"[FETCH] {path}")
                downloaded += 1
            else:
                print(f"[ERROR] {path}: {msg}")
                errors += 1

    print(f"\nSummary: downloaded={downloaded}, skipped={skipped}, errors={errors}")
    if OUT_DIR.exists():
        actual_files = list(OUT_DIR.rglob("*"))
        print(f"Actual files in output: {len(actual_files)}")


if __name__ == "__main__":
    main()
