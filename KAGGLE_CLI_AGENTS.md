# Connecting an Agent to Kaggle — Kaggle CLI Only

**Audience:** any agent (or human) that needs to run code on Kaggle's servers and get the results back, using **nothing but the `kaggle` CLI**. No web UI clicks, no browser, no direct calls to Kaggle's internal Jupyter proxy.

**Status of this document:** every command and every claim below was executed for real against Kaggle in this workspace. Where something is a caveat rather than a verified fact, it is marked as such.

---

## 1. The mental model

Think of the CLI as a courier service with three objects:

| Object                   | What it is                                                                                                         | Where it lives                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------ |
| **Token**          | Your account's API key. Proves you are you.                                                                        | Your environment, one variable |
| **Kernel package** | A folder holding your code plus a `kernel-metadata.json` manifest ("luggage tag": owner, name, runtime settings) | Your disk                      |
| **Kernel version** | An immutable snapshot of that package, queued and executed on Kaggle hardware                                      | Kaggle's servers               |

The whole loop is four verbs:

```
push  ──►  status  ──►  logs  ──►  output
(send &   (wait for    (read     (download
 start)    finish)      stdout)   artifacts)
```

There is **no interactive shell** on the Kaggle side. You cannot "ssh in". Code runs as a batch job: it starts, prints, writes files, exits. Therefore the reliable pattern is:

> **Write results to files, not just to stdout.** stdout can be truncated or hard to parse; downloaded files are durable.

---

## 2. Prerequisites

```bash
pip install kaggle        # provides the `kaggle` executable
kaggle --version          # expect "Kaggle CLI 2.x"
```

Get an API token from **https://www.kaggle.com/settings/api** → *Generate New Token*. Newer tokens look like `KGAT_<hex>`. The legacy `kaggle.json` (`{"username":..., "key":...}`) also still works.

---

## 3. Step 1 — Authenticate without a browser

This is the part that trips agents up: `kaggle auth login` opens a **browser** and waits for a human. Do not call it from an agent. Use the token instead.

### The agent-safe way: environment variable, per command

```powershell
# Windows / PowerShell
$env:KAGGLE_API_TOKEN = "<token>"
kaggle config view
```

```bash
# Linux / macOS / bash
export KAGGLE_API_TOKEN="<token>"
kaggle config view
```

Expected healthy output:

```
Configuration values from C:\Users\<you>\.kaggle
- username: <your-username>
- auth_method: ACCESS_TOKEN
```

`auth_method: ACCESS_TOKEN` means the token was picked up. If you instead get *"Authentication required to call the Kaggle API"*, the variable is wrong, empty, or not visible to the child process.

### Hugging Face uploads are separate

Kaggle authentication does not give access to Hugging Face. Any kernel that uploads to Hugging Face reads the token from a private Kaggle dataset that contains a text file with the token. The dataset is attached via `dataset_sources` in `kernel-metadata.json` and mounted under `/kaggle/input/`. The code reads the token from that mounted file. Never hardcode the token and never put it in a repository file.

#### Private dataset workaround for Hugging Face tokens

1. Create a local folder, e.g. `secret-folder/`.
2. Inside it, create `hf_token.txt` containing only your Hugging Face write token.
3. Upload it as a private dataset:
   ```powershell
   kaggle datasets create -p ./secret-folder --private
   ```
   Note the dataset slug printed by Kaggle, e.g. `deep-research-1a10/hf-token-store`.
4. Attach that dataset in `kernel-metadata.json`:
   ```json
   "dataset_sources": ["deep-research-1a10/hf-token-store"]
   ```
5. In your Python code, read the token from the mounted path:
   ```python
   with open("/kaggle/input/datasets/deepresearch1a10/hf-token-store/hf_token.txt") as f:
       HF_TOKEN = f.read().strip()
   ```
6. Datasets persist across pushes, so you do not need to reattach them when you push new kernel versions.


### Rules that matter

1. **Set the variable in the same shell invocation as the command.** Sandboxes, containers, and `pwsh -Command` calls frequently start a fresh process per command, so an `export` from a previous call may not survive.
2. **Never write the token into a repository file** (no `.env` committed, no token in `kernel-metadata.json`, no token in code you `push`). Kaggle kernels can read their own files, and `kernel_sources`/`dataset_sources` are shared surfaces.
3. **Prefer the env var over `~/.kaggle/access_token`** when running under a file sandbox that only permits writes inside the workspace — writing to the home directory may be denied, and the failure looks like an auth failure.
4. **Tokens are account-wide.** Treat them like a password: rotate at `kaggle.com/settings/api` if leaked.

### Cheap auth + quota check

```bash
kaggle quota
```

```
resource  used   remaining  total   refreshAt
GPU       0.00h  30.00h     30.00h  2026-10-31T00:00:00+00:00
TPU       0.00h  20.00h     20.00h  2026-10-31T00:00:00+00:00
```

This is the single best smoke test: it proves the token works **and** tells you what accelerator budget you have left.

---

## 4. Step 2 — Build the kernel package

A kernel package is a folder:

```
my_kernel/
├── kernel-metadata.json     # required, exact filename
└── train.py                 # your code (script kernel) or analysis.ipynb (notebook kernel)
```

### `kernel-metadata.json` for a script kernel

```json
{
  "id": "your-username/my-kernel-slug",
  "title": "My Kernel",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

### Field reference

| Field                   | Required      | Notes                                                                                                                                                               |
| ----------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                  | **yes** | `owner/slug`. Must match your authenticated username. Creating a *new* slug runs it; re-using an existing one creates **version N+1** of the same kernel. |
| `title`               | yes           | Human-readable.                                                                                                                                                     |
| `code_file`           | **yes** | Path to your code file *inside* the folder. For `kernel_type: "notebook"` this is the `.ipynb`.                                                               |
| `language`            | yes           | `"python"`, `"r"`, `"julia"`, `"sqlite"`.                                                                                                                   |
| `kernel_type`         | yes           | `"script"` (batch `.py`/`.r`) or `"notebook"` (`.ipynb`).                                                                                                 |
| `is_private`          | recommended   | **Set `true`.** Public kernels are visible to the whole site.                                                                                               |
| `enable_gpu`          | no            | Legacy boolean. Prefer the `--accelerator` flag on `push` (see §5).                                                                                            |
| `enable_internet`     | no            | **Defaults to off.** If your code `pip install`s anything or downloads data at runtime, you must set `true`.                                              |
| `dataset_sources`     | no            | `["owner/dataset-slug"]` → mounted read-only under `/kaggle/input/`.                                                                                           |
| `competition_sources` | no            | `["competition-slug"]` → competition data under `/kaggle/input/`. Accepting rules on the site may be a prerequisite.                                           |
| `kernel_sources`      | no            | Other kernels' outputs, e.g. `["owner/kernel-slug"]`.                                                                                                             |
| `model_sources`       | no            | `["owner/model/framework/variation/version"]`.                                                                                                                    |

### Where does my code run, and where do I write?

| Path                | Meaning                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `/kaggle/working` | **Writable.** The only place whose contents come back via `kaggle kernels output`. Its starting cwd. |
| `/kaggle/input`   | Read-only mounts: datasets, competition data, attached kernel outputs, models.                               |
| `/tmp`            | Writable scratch, **not** returned.                                                                    |

Practical rule for the code itself:

```python
import os
out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
```

so the same script also runs locally without edits.

---

## 5. Step 3 — Push (upload and start the run)

```bash
kaggle kernels push -p ./my_kernel
```

Success looks like:

```
Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/<user>/<slug>
```

That means **queued and started**, not finished. Push returns quickly (seconds).

Useful flags:

| Flag                            | Effect                                                                                                           |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `-p, --path FOLDER`           | The package folder. Defaults to cwd — always pass it explicitly so a stray `cd` cannot push the wrong folder. |
| `--accelerator NvidiaTeslaT4` | Run on GPU (T4 x2). Other values include `TpuV5E8`. Omit for CPU.                                              |
| `-t, --timeout SECONDS`       | Cap the run duration; Kaggle's own maximum still applies.                                                        |
| `--no-run`                    | Save a new version **without** executing it.                                                               |

Notes:

- **Re-pushing the same `id` creates version N+1** — the previous version's outputs stay intact under that version number, which makes A/B comparisons easy.
- Kernels are single-use. Once a run completes or is cancelled, create a new kernel slug for the next run instead of reusing the same slug.
- A warning like `The following are not valid tags and could not be added` is **cosmetic**; the run still proceeds. Only an error mentioning `id`, auth, or metadata is fatal.
- GPU runs may require a **phone-verified** Kaggle account. If a GPU push fails with a 403/permission message rather than a queue message, that is usually the cause — retry without `--accelerator`, or verify the phone on the site.

---

## 6. Step 4 — Wait for the run

```bash
kaggle kernels status <owner>/<slug>
```

```
<owner>/<slug> has status "KernelWorkerStatus.COMPLETE"
```

### The single most important gotcha

**`kaggle kernels status` exits with code 0 regardless of the run's outcome.** A crashed kernel still gives exit code 0 because the *CLI call* succeeded. Never treat return-code 0 as "the code worked".

Parse the string instead. States you will see are uppercase enum names:

| Status                          | Meaning                          | Agent action                                              |
| ------------------------------- | -------------------------------- | --------------------------------------------------------- |
| `KernelWorkerStatus.QUEUED`   | Waiting for a machine            | keep polling                                              |
| `KernelWorkerStatus.RUNNING`  | Executing                        | keep polling                                              |
| `KernelWorkerStatus.COMPLETE` | Finished normally                | fetch logs + outputs, then **still** verify content |
| `KernelWorkerStatus.ERROR`    | Your code raised / crashed       | fetch logs, read traceback                                |
| `KernelWorkerStatus.CANCELED` | Stopped (timeout, manual, quota) | inspect logs for the reason                               |

A poll loop that behaves (bash):

```bash
for i in $(seq 1 60); do
  s=$(kaggle kernels status "$REF" 2>&1)
  echo "$s"
  case "$s" in
    *COMPLETE*|*ERROR*|*CANCELED*) break ;;
  esac
  sleep 15
done
```

Timing observed: a trivial PyTorch training kernel went `RUNNING → COMPLETE` in roughly **40 seconds**, of which about 12 seconds was actual model work — machine spin-up, image pull and notebook conversion dominate small jobs. Budget minutes, not seconds, for the first run after a while.

---

## 7. Step 5 — Read the logs

```bash
kaggle kernels logs <owner>/<slug>          # snapshot
kaggle kernels logs <owner>/<slug> -f       # follow live, like tail -f
```

The output is **JSON-lines**, one record per write:

```json
[{"stream_name":"stdout","time":6.10,"data":"python      : 3.13.15\n"}
,{"stream_name":"stdout","time":6.20,"data":"parameter count: 1,059,841\n"}]
```

Parse with `json` after wrapping in `[]`, or just grep the `data` substrings. Expect a block of `stderr` noise from Kaggle's own tooling (`nbconvert`, `mistune` `SyntaxWarning: invalid escape sequence`) on **every** run, including successful ones — these are not your bug and not a failure signal.

This is also your **proof of where the code ran**. A healthy log begins with something like:

```
python      : 3.13.15
platform    : Linux-6.18.55+-x86_64-with-glibc2.39
torch       : 2.11.0+cpu
cwd         : /kaggle/working
```

If you see `Windows` or your local `C:\...` path there, you did not run on Kaggle. Log the environment explicitly — it costs three lines and makes "did this actually happen remotely?" answerable.

---

## 8. Step 6 — Retrieve the artifacts

```bash
kaggle kernels files  <owner>/<slug>                     # list what exists (free)
kaggle kernels output <owner>/<slug> -p ./out            # download to ./out
```

Example listing:

```
name          size  creationDate
results.json   869  2:37 pm, Saturday 24 October 2026 UTC
results.txt    865  2:37 pm, Saturday 24 October 2026 UTC
```

Options: `-o/--force` (re-download), `-q/--quiet`, `--file-pattern REGEX` (fetch only what you need — important when a run emits gigabytes of checkpoints).

Outputs are uploaded to Kaggle's kernel output storage. **Because they persist there, a later run is not the only copy** — you can re-download a previous version with `kaggle kernels output <owner>/<slug>/<version>`.

### Windows-specific trap that looks like a failure

On PowerShell, `kaggle`'s progress bars are written to **stderr**. PowerShell treats that as an error condition and reports `[exit code: 1]` — even though every file downloaded correctly. Observed verbatim:

```
0%|          | 0.00/332 [00:00<?, ?B/s]
100%|##########| 332/332 [00:00<00:00, 65.7kB/s]
...
[exit code: 1]
```

**Mitigation:** judge success by *files present and non-empty*, not by exit code. Or wrap with `2>$null` / append `| Out-String`. Do not "retry the download" in a loop and conclude Kaggle is broken.

---

## 9. Verify, don't assume

A run is only trustworthy when all four hold:

1. `status` string contains `COMPLETE`.
2. `logs` exist and their first lines show a **Kaggle Linux** environment.
3. `logs` end with your own completion marker (e.g. `million_param_training_done`), and contain no Python traceback.
4. The expected output files downloaded and contain plausible values (e.g. loss decreasing, accuracy non-zero, row count > 0).

Check 4 is the one agents skip. A kernel can be `COMPLETE` while having written an empty or NaNs-only file because your script swallowed an exception with `try/except`.

---

## 10. Copy-paste: minimal smoke-test kernel

Use this to prove the pipeline works before pushing real work. It costs about a minute of wall time and no GPU quota.

`smoke/kernel-metadata.json`

```json
{
  "id": "<your-username>/agent-smoke-test",
  "title": "Agent Smoke Test",
  "code_file": "hello.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
```

`smoke/hello.py`

```python
import os, platform, sys, socket

print("host      :", socket.gethostname())
print("python    :", sys.version.split()[0])
print("platform  :", platform.platform())
print("cwd       :", os.getcwd())
print("AGENT-REMOTE-PROOF-OK")

with open("/kaggle/working/proof.txt", "w") as f:
    f.write("remote execution confirmed on %s\n" % platform.platform())
```

Then:

```bash
$env:KAGGLE_API_TOKEN = "<token>"      # bash: export KAGGLE_API_TOKEN=...
kaggle kernels push -p ./smoke
kaggle kernels status <your-username>/agent-smoke-test
kaggle kernels logs   <your-username>/agent-smoke-test
kaggle kernels output <your-username>/agent-smoke-test -p ./smoke/out
```

Note the contrast with the expired-token scripts previously in this workspace (`run_kaggle_*.py`): those spoke directly to a notebook's Jupyter proxy at `kkb-production.jupyter-proxy.kaggle.net` using a session JWT. That route is undocumented, dies with the session, and binds you to one browser-opened notebook. **The CLI route above is the supported one.**

---

## 11. Failure modes cheat-sheet

| Symptom                                                   | Cause                                                                                      | Fix                                                                                           |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `Authentication required to call the Kaggle API`        | Token absent/wrong, or set in a different process                                          | Re-set `KAGGLE_API_TOKEN` in the same command invocation; verify via `kaggle config view` |
| `403` / permission on push                              | `id` owner ≠ authenticated username                                                     | Use the username printed by `kaggle config view`                                            |
| `403` mentioning phone / GPU quota                      | GPU needs a verified account                                                               | Run without `--accelerator`, or verify phone on the site                                    |
| `The following are not valid tags ...`                  | Unknown keywords in metadata                                                               | Harmless; remove the `keywords` field to silence it                                         |
| Push works, status goes `ERROR` fast                    | Your script raised before doing anything                                                   | `kaggle kernels logs`, read the traceback                                                   |
| Status `ERROR` but logs look fine                       | Error occurred during notebook conversion/packaging                                        | Look at the tail of logs, especially `stderr`                                               |
| `COMPLETE` but no output files                          | Wrote to cwd `/kaggle/working`? No → wrote to `/tmp` or `/kaggle/input` (read-only) | Write your artifacts explicitly to `/kaggle/working`                                        |
| `COMPLETE`, output files empty                          | Exception swallowed by `try/except`, or wrong path                                       | Fail loudly: re-raise, or write a `status: failed` marker                                   |
| `ModuleNotFoundError` for a packaged lib                | `enable_internet` false and dependency not preinstalled                                  | Set `enable_internet: true` + `pip install` in-code, or attach it as a dataset            |
| Kernel killed near the end                                | Max runtime (typically 12h; 9h GPU) or disk (≈20GB)                                       | Reduce work, checkpoint to outputs, or shard across runs                                      |
| PowerShell shows `[exit code: 1]` after a good download | stderr progress bars                                                                       | Check files instead of exit code                                                              |

---

## 12. Quota discipline

- `kaggle quota` before and after any GPU/TPU job.
- GPU runs bill **wall-clock**, including image pull and idle setup — a 2-minute job still costs minutes.
- Prefer CPU for anything that finishes in seconds; save GPU for actual training.
- `--no-run` when you only want to update a kernel's code/prose without burning a run.

---

## 13. One-paragraph summary for an agent's system prompt

> Authenticate with `KAGGLE_API_TOKEN` in the same process as the command, ideally as `$env:KAGGLE_API_TOKEN` / `export KAGGLE_API_TOKEN`. Build a folder with `kernel-metadata.json` (`id: <user>/<slug>`, `code_file`, `kernel_type: script`, `is_private: true`) plus your `.py`. `kaggle kernels push -p <folder>`. Poll `kaggle kernels status <user>/<slug>` until the string contains `COMPLETE`, `ERROR`, or `CANCELED` — the exit code is always 0 and proves nothing. Read `kaggle kernels logs <user>/<slug>` (JSON-lines; ignore nbconvert `SyntaxWarning` noise) and confirm the header shows Linux and `/kaggle/working`. Download artifacts with `kaggle kernels output <user>/<slug> -p <dir>`, and verify their contents rather than trusting exit codes. Always write results to `/kaggle/working`, and put a final marker line in stdout so success is unambiguous.

---

also make sure to use KAGGLE CPU for light tasks and when user says so and USE GPU when user says so and TPU only when confirmed by user twice . CPU can be used without permission and for most general tasks . Device is selected based on code requirement such that same code can be run on different machine depending on either it is being only tested or it is a full run . It is truly possible to use CPU for a gpu full code when testing only for small scale or testing a cpu part .
