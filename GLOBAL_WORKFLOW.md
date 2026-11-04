# Global Workflow Guide

## Purpose

This guide describes a reusable way of working that can be used by any repository. The code lives in a local working copy on one computer. A watcher keeps an online code site in sync with that local copy. Heavy work runs in an execution environment such as Kaggle. Large files and results are stored in a separate storage site such as Hugging Face.

## Architecture

- Local working copy: where all edits happen.
- Online code site: source of truth for code, updated by the watcher.
- Execution environment: runs the code, such as Kaggle for GPU or CPU work.
- Storage site: holds large files, results, and shared data, such as Hugging Face.

## How it works

### 1. Code changes

- All edits happen in the local working copy.
- The watcher checks the local copy at a fixed interval.
- If the local copy has changes, the watcher waits one cycle to make sure the changes are finished and stable.
- If the changes are still the same, the watcher sends them to the online code site.
- The watcher never brings changes from the online code site down to the local copy.
- If both sides change at the same time, the watcher stops and asks for help.

### 2. Execution

- The execution environment downloads the latest code from the online code site.
- It installs dependencies.
- It runs the requested command.
- It uploads results to the storage site.

### 3. Results storage

- Large outputs must be uploaded to the storage site. This step is required, not optional.
- Code and small files stay in the online code site.
- The local copy can sync large files from the storage site when needed.
- Results must not be left only on the execution environment or only on one local computer.

## Watcher behavior

- Runs in the background all the time.
- Checks at a fixed interval.
- Waits for stable changes before sending.
- Stages all changes, commits with a timestamp, and pushes.
- Logs every action.
- Uses a lock file so only one copy runs.
- Stops cleanly when asked.
- Retries on failure.

## How to make changes

### For the agent

- Read and edit files in the local working copy.
- Let the watcher send changes.
- Do not edit the online code site directly.

### For the user

- Edit files in the local working copy.
- Or ask the agent to make changes.
- The watcher sends changes automatically.

### For the execution environment

- Re-run the relevant job.
- It downloads the latest code from the online code site.
- No manual copying is needed.
- Kaggle kernels are single-use. After a run finishes or is cancelled, create a new kernel rather than pushing again to the same slug.

## Important notes

- Local working copy is the editing surface.
- Watcher is the only path from local to the online code site.
- Watcher never pulls remote changes down.
- Execution environment always gets code from the online code site.
- Storage site is the source of truth for large files and results.
- Upload of results to the storage site is mandatory.
- Storage site uploads need a storage token: execution environments upload to the storage site using a token stored in a private dataset. The dataset contains a single text file with the token, and the code reads that file from the mounted dataset path, for example `/kaggle/input/datasets/deepresearch1a10/hf-token-store/hf_token.txt`.
- If the watcher is not running, local changes stay local.
- If both sides change at once, the watcher stops and asks for help.

## Setup steps

1. Clone the online code site to create the local working copy.
2. Configure login to the online code site.
3. Start the watcher once.
4. Leave the watcher running.
5. Make edits in the local working copy.

## Repository structure (generic)

- Main code folder
- Tests folder
- Execution environment folder with thin wrappers
- Watcher program
- Watcher start file
- Watcher stop file
- Guide files
- Ignore list for temporary and generated files