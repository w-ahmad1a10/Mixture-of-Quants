"""Watch the local repository and push local changes to the online code site.

The watcher never pulls changes from the online code site. It only sends
local changes outward, every 60 seconds, when they are stable.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCK_FILE = ROOT / "watcher.lock"
LOG_FILE = ROOT / "watcher.log"
BRANCH = "main"
REMOTE = "origin"
INTERVAL_SECONDS = 60


def log(message: str) -> None:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    line = f"{timestamp} {message}\n"
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def run_command(command: list[str], timeout: int = 60) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def git_status() -> tuple[str | None, str]:
    code, output, error = run_command(["git", "status", "--porcelain"])
    if code != 0:
        return None, error
    return output, ""


def acquire_lock() -> bool:
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = None
        if pid is not None:
            try:
                os.kill(pid, 0)
                return False
            except OSError:
                pass
        try:
            LOCK_FILE.unlink()
        except OSError:
            pass
    try:
        LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        return False
    return True


def release_lock() -> None:
    try:
        if LOCK_FILE.exists() and LOCK_FILE.read_text(encoding="utf-8").strip() == str(
            os.getpid()
        ):
            LOCK_FILE.unlink()
    except OSError:
        pass


def signature_for(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch local changes and push them.")
    parser.add_argument(
        "--interval",
        type=int,
        default=INTERVAL_SECONDS,
        help="Seconds between checks.",
    )
    arguments = parser.parse_args()
    interval = max(5, arguments.interval)

    if not acquire_lock():
        print("Another watcher is already running.")
        return 1

    log("watcher started")
    last_signature: str | None = None
    try:
        while True:
            status_text, error = git_status()
            if status_text is None:
                log(f"git status failed: {error}")
                time.sleep(interval)
                continue

            if status_text == "":
                last_signature = None
                run_command(["git", "fetch", REMOTE, BRANCH], timeout=30)
                code, ahead, _ = run_command(
                    ["git", "rev-list", "--count", f"HEAD..{REMOTE}/{BRANCH}"]
                )
                if code == 0 and ahead not in ("", "0"):
                    log(
                        f"remote has {ahead} new commits; not pulling (local-only policy)"
                    )
                time.sleep(interval)
                continue

            current_signature = signature_for(status_text)
            if last_signature != current_signature:
                last_signature = current_signature
                log("local changes detected; waiting for stability")
                time.sleep(interval)
                continue

            log("changes stable; syncing to online code site")
            fetch_code, _, fetch_error = run_command(
                ["git", "fetch", REMOTE, BRANCH], timeout=30
            )
            if fetch_code != 0:
                log(f"fetch failed: {fetch_error}; will retry")
                time.sleep(interval)
                continue

            code, ahead, _ = run_command(
                ["git", "rev-list", "--count", f"HEAD..{REMOTE}/{BRANCH}"]
            )
            if code == 0 and ahead not in ("", "0"):
                log(
                    f"conflict: remote is {ahead} commits ahead while local has changes; "
                    "stopping watcher"
                )
                break

            MAX_FILE_BYTES = 5 * 1024 * 1024
            too_large = []
            for line in status_text.splitlines():
                path = line[3:].strip() if len(line) > 3 else ""
                if " -> " in path:
                    path = path.split(" -> ", 1)[1].strip()
                if path:
                    full_path = os.path.join(ROOT, path)
                    try:
                        if os.path.isfile(full_path) and os.path.getsize(full_path) > MAX_FILE_BYTES:
                            too_large.append(path)
                    except OSError:
                        pass
            if too_large:
                log(f"size guard: files larger than 5 MB: {', '.join(too_large)}; skipping sync")
                time.sleep(interval)
                continue

            add_code, _, add_error = run_command(["git", "add", "-A"])
            if add_code != 0:
                log(f"git add failed: {add_error}")
                time.sleep(interval)
                continue

            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            commit_message = f"auto: sync local changes {timestamp}"
            commit_code, _, commit_error = run_command(
                ["git", "commit", "-m", commit_message]
            )
            if commit_code != 0:
                log(f"commit returned {commit_code}: {commit_error}")
                time.sleep(interval)
                continue

            push_code, _, push_error = run_command(
                ["git", "push", REMOTE, BRANCH], timeout=60
            )
            if push_code != 0:
                log(f"push failed: {push_error}; will retry")
                time.sleep(interval)
                continue

            log("sync complete")
            last_signature = None
            time.sleep(interval)
    except KeyboardInterrupt:
        log("watcher stopped by user")
    finally:
        release_lock()

    return 0


if __name__ == "__main__":
    sys.exit(main())
