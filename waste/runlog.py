"""
runlog.py — Local-only run logger. Replaces wandb entirely.
Repos ONLY write files here; the dashboard does everything else.

    from runlog import RunLog
    run = RunLog("glu-silu-100L", config={"lr": 3e-4})   # creates Logs/<name>/
    run.log({"loss": 4.2, "lr": 3e-4}, step=100)
    run.log_table("grad_rms", columns=["tensor","rms"], rows=[[...],[...]])
    run.finish()

Writes: Logs/<run_name>/metrics.jsonl , config.yaml , tables/<name>.csv
"""
import csv, json, os, time
from datetime import datetime


class RunLog:
    def __init__(self, run_name=None, config=None, base_dir=None):
        if base_dir is None:
            base_dir = os.path.join("Logs")
        run_name = run_name or "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        # unique-ify
        self.run_dir = os.path.join(base_dir, run_name)
        i = 2
        while os.path.exists(self.run_dir):
            self.run_dir = os.path.join(base_dir, f"{run_name}_{i}")
            i += 1
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "tables"), exist_ok=True)

        self.metrics_file = os.path.join(self.run_dir, "metrics.jsonl")
        self._step = 0
        self._t0 = time.time()
        self._finished = False

        cfg = dict(config or {})
        cfg.setdefault("_meta", {})
        cfg["_meta"].update({
            "run_name": os.path.basename(self.run_dir),
            "started_utc": datetime.utcnow().isoformat() + "Z",
            "logger": "runlog/1.0 (local dashboard system)",
        })
        with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
            try:
                import yaml
                yaml.safe_dump(cfg, f, default_flow_style=False)
            except Exception:
                json.dump(cfg, f, indent=2, default=str)
        print(f"[runlog] logging to {self.run_dir}")

    # ---- core ----
    def log(self, metrics: dict, step: int | None = None):
        if self._finished:
            return
        self._step = self._step + 1 if step is None else step
        entry = {"step": self._step,
                 "timestamp": time.time(),
                 **{k: v for k, v in metrics.items()}}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_table(self, name: str, columns: list, rows: list):
        """Replaces wandb.Table. Saved as CSV next to metrics; dashboard shows it."""
        path = os.path.join(self.run_dir, "tables", f"{name}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(columns)
            w.writerows(rows)

    def log_artifact(self, local_path: str, copy_into_run: bool = True):
        """Register a file produced by the run. Small files are copied into the run dir."""
        if copy_into_run and os.path.getsize(local_path) < 50 * 1024 * 1024:
            import shutil
            shutil.copy2(local_path, os.path.join(self.run_dir, os.path.basename(local_path)))
        with open(os.path.join(self.run_dir, "artifacts.txt"), "a") as f:
            f.write(local_path + "\n")

    def finish(self):
        if not self._finished:
            self.log({"_runtime_seconds": time.time() - self._t0, "_finished": True})
            self._finished = True
            print(f"[runlog] finished — view in dashboard: {self.run_dir}")

    # ---- wandb-style aliases so old code needs minimal edits ----
    def init(self, *a, **k):          # wandb.init(...) -> already done in ctor
        return self

    @staticmethod
    def Table(columns, rows):         # wandb.Table(...) -> pass straight to log_table
        return {"columns": list(columns), "rows": [list(r) for r in rows]}
