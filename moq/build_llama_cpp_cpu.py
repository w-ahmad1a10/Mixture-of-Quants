#!/usr/bin/env python3
"""
Build llama.cpp CPU Binaries

Downloads and sets up the latest llama.cpp CPU binaries for Ubuntu x64.
Fetches the latest release from GitHub, downloads the appropriate asset, and extracts it.

CLI Arguments:
  --output-dir          Directory where the extracted binaries are placed
                        (default: /workspace/llama_bin)
  --repo                GitHub repo to fetch releases from
                        (default: ggml-org/llama.cpp)
  --source-dir          Directory where the llama.cpp source tree is cloned for
                        conversion scripts (default: /workspace/llama.cpp)
  --system-lib-dir      System directory to copy shared .so libraries into
                        (default: /usr/lib)
  --skip-download       Skip downloading the tarball if it already exists in /tmp
  --install-system-libs Copy shared libraries to --system-lib-dir and run ldconfig
  --no-clone-source     Skip cloning the llama.cpp source repo after binary extraction

Usage examples:
  python build_llama_cpp_cpu.py
  python build_llama_cpp_cpu.py --output-dir /workspace/llama_bin
  python build_llama_cpp_cpu.py --output-dir /tmp/llama --install-system-libs
  python build_llama_cpp_cpu.py --skip-download --no-clone-source
"""

import os
import subprocess
import json
import argparse
from pathlib import Path


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=capture_output,
        text=True
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result.stdout.strip() if capture_output else None


def find_latest_release(repo):
    print(f"🔍 Using pinned release b8121 for {repo}...")
    build = "b8121"
    # Fetch release info from GitHub API
    result = subprocess.run(
        ["curl", "-s", f"https://api.github.com/repos/{repo}/releases/tags/{build}"],
        capture_output=True, text=True
    )
    latest = json.loads(result.stdout)
    print(f"✅ Release: {build}")
    return latest, build


def find_ubuntu_asset(assets, repo, build):
    """Find the correct Ubuntu x64 asset (no CUDA/Vulkan/OpenVINO)."""
    asset_names = [a["name"] for a in assets]
    print(f"   Available assets: {[a for a in asset_names if 'ubuntu' in a.lower()]}")

    exact_match_file = f"llama-{build}-bin-ubuntu-x64.tar.gz"
    if exact_match_file in asset_names:
        return exact_match_file

    for asset in assets:
        name = asset["name"]
        if ("ubuntu" in name.lower() and
            "x64" in name.lower() and
            "cuda" not in name.lower() and
            "vulkan" not in name.lower() and
            "openvino" not in name.lower()):
            return name

    raise RuntimeError("No suitable Ubuntu x64 asset found")


def download_file(url, output_path):
    """Download a file with progress."""
    cmd = f'wget --show-progress "{url}" -O "{output_path}"'
    print(f"Downloading: {url}")
    run_command(cmd, check=True)


def extract_tarball(tarball_path, extract_dir):
    """Extract tarball to directory."""
    os.makedirs(extract_dir, exist_ok=True)
    run_command(f'tar -xzf "{tarball_path}" -C "{extract_dir}"', check=True)


def fix_nested_structure(extract_dir):
    """Fix nested folder structure from extraction."""
    run_command(
        f'find {extract_dir} -mindepth 2 -type f -name "*.so*" '
        f'-exec mv -n {{}} {extract_dir}/ \\;'
    )
    run_command(
        f'find {extract_dir} -mindepth 2 -type f -name "llama-*" '
        f'-exec mv -n {{}} {extract_dir}/ \\;'
    )
    run_command(f'chmod +x {extract_dir}/llama-*')


def install_system_libs(extract_dir, system_lib_dir):
    """Copy shared libraries to system directory."""
    run_command(f'cp -n {extract_dir}/*.so* {system_lib_dir} 2>/dev/null || true')
    run_command('ldconfig 2>/dev/null || true')


def verify_installation(extract_dir):
    """Verify that required binaries exist."""
    binaries = ['llama-quantize', 'llama-imatrix']
    missing = []
    for binary in binaries:
        path = os.path.join(extract_dir, binary)
        if not os.path.exists(path):
            missing.append(binary)
    if missing:
        print(f"\n⚠️  Some binaries not found: {missing}")
        return False
    print(f"\n✅ SETUP COMPLETE — llama-imatrix and llama-quantize are ready")
    return True


def run_build_llama(args):
    print(f"\n=== llama.cpp CPU Build ===")
    print(f"Repo       : {args.repo}")
    print(f"Output dir : {args.output_dir}")
    print(f"Source dir : {args.source_dir}")

    latest, build = find_latest_release(args.repo)
    assets = latest.get('assets', [])
    try:
        filename = find_ubuntu_asset(assets, args.repo, build)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("\nAll available assets:")
        for a in assets:
            print(f"  - {a['name']}")
        return 1

    print(f"✅ Downloading: {filename}")
    url = f"https://github.com/{args.repo}/releases/download/{build}/{filename}"
    tarball_path = os.path.join('/tmp', filename)

    if not args.skip_download or not os.path.exists(tarball_path):
        download_file(url, tarball_path)
    else:
        print(f"Using existing tarball: {tarball_path}")

    if not os.path.exists(tarball_path) or os.path.getsize(tarball_path) < 1000:
        raise RuntimeError(f"❌ Download failed or file too small: {filename}")
    print(f"✅ Downloaded {os.path.getsize(tarball_path)/1e6:.1f} MB")

    print("\n=== Extracting ===")
    extract_tarball(tarball_path, args.output_dir)
    fix_nested_structure(args.output_dir)

    if args.install_system_libs:
        print("\n=== Installing System Libraries ===")
        install_system_libs(args.output_dir, args.system_lib_dir)

    print("\n=== Verification ===")
    print("Binaries extracted:")
    run_command(f'ls {args.output_dir}/llama-* | head -20')

    success = verify_installation(args.output_dir)

    if not args.no_clone_source and not os.path.exists(args.source_dir):
        print(f"\n📥 Cloning llama.cpp source to {args.source_dir}...")
        run_command(f'git clone https://github.com/{args.repo} {args.source_dir}')
        print(f"✅ Source cloned")

    print("\n✅ SETUP COMPLETE. Binaries and scripts are ready.")
    print(f"   Binaries: {args.output_dir}")
    print(f"   Source:   {args.source_dir}")

    return 0 if success else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build and setup llama.cpp CPU binaries.')
    parser.add_argument('--output-dir', default='/workspace/llama_bin',
                        help='Directory where extracted binaries are placed (default: /workspace/llama_bin)')
    parser.add_argument('--repo', default='ggml-org/llama.cpp',
                        help='GitHub repo to fetch releases from (default: ggml-org/llama.cpp)')
    parser.add_argument('--source-dir', default='/workspace/llama.cpp',
                        help='Directory to clone the llama.cpp source tree into (default: /workspace/llama.cpp)')
    parser.add_argument('--system-lib-dir', default='/usr/lib',
                        help='System directory for shared .so libraries (default: /usr/lib)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloading the tarball if it already exists in /tmp')
    parser.add_argument('--install-system-libs', action='store_true',
                        help='Copy shared libraries to --system-lib-dir and run ldconfig')
    parser.add_argument('--no-clone-source', action='store_true',
                        help='Skip cloning the llama.cpp source repo after binary extraction')
    args = parser.parse_args()
    exit(run_build_llama(args))
