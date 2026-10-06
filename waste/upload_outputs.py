#!/usr/bin/env python3
"""Upload MoQ outputs to HF dataset repo.

Usage:
  python upload_outputs.py <local_path> <hf_path_in_repo>

Examples:
  python upload_outputs.py results/calib/part-0.parquet goal2/calib/part-0.parquet
  python upload_outputs.py logs/goal1/run-1.txt goal1/logs/run-1.txt
  python upload_outputs.py outputs/folder/ goal2/outputs/
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "w-ahmad/moq-lfm25-230m-outputs"
TOKEN_PATH = "/home/waleed/.cache/huggingface/token"


def load_token() -> str:
    token_path = Path(TOKEN_PATH)
    if not token_path.exists():
        raise FileNotFoundError(f"HF token not found at {TOKEN_PATH}")
    token = token_path.read_text().strip()
    if not token:
        raise ValueError("HF token file is empty")
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload MoQ outputs to HF dataset repo")
    parser.add_argument("local_path", help="Local file or folder to upload")
    parser.add_argument("hf_path", help="Destination path inside the HF dataset repo")
    args = parser.parse_args()

    local_path = Path(args.local_path)
    if not local_path.exists():
        print(f"ERROR: local path does not exist: {local_path}", file=sys.stderr)
        sys.exit(1)

    token = load_token()
    api = HfApi(token=token)

    print(f"Repo : {REPO_ID}")
    print(f"Local: {local_path}")
    print(f"HF   : {hf_path}")

    if local_path.is_file():
        print("Mode : single file")
        url = api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=hf_path,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print(f"Uploaded: {url}")
    elif local_path.is_dir():
        print("Mode : folder")
        # Strip trailing slash to avoid issues
        hf_path = hf_path.rstrip("/")
        url = api.upload_folder(
            folder_path=str(local_path),
            path_in_repo=hf_path,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print(f"Uploaded: {url}")
    else:
        print(f"ERROR: unsupported path type: {local_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
