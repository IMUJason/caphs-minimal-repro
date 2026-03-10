from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SWIFT_DIR = ROOT / "swift_bench"
WORKLOAD_DIR = ROOT / "workloads"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw_runs"
PROFILE_PATH = RESULTS_DIR / "profile_m4_real.json"
RUN_REGISTER = RESULTS_DIR / "run_register.csv"
MONITOR_DIR = RESULTS_DIR / "monitoring"
POLICIES = ["fcfs", "least_loaded", "fixed_priority", "work_stealing", "caphs"]


def build_binary() -> Path:
    subprocess.run(["swift", "build", "-c", "release"], cwd=SWIFT_DIR, check=True)
    return SWIFT_DIR / ".build" / "release" / "M4HeteroBench"


def ensure_profile(binary: Path) -> None:
    if PROFILE_PATH.exists():
        return
    subprocess.run(
        [
            str(binary),
            "--mode", "profile",
            "--sizes", "768,1024,1536,2048",
            "--reps", "5",
            "--output", str(PROFILE_PATH),
        ],
        check=True,
        cwd=ROOT,
        env={**os.environ, "VECLIB_MAXIMUM_THREADS": "1"},
    )


def read_top_stream(proc: subprocess.Popen, samples: list[dict], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        line = proc.stdout.readline()
        if not line:
            break
        text = line.strip()
        if not text or not text[0].isdigit():
            continue
        parts = text.split()
        if len(parts) < 5:
            continue
        try:
            samples.append(
                {
                    "timestamp": time.time(),
                    "pid": int(parts[0]),
                    "cpu_pct": float(parts[2]),
                    "mem": parts[3],
                    "power": float(parts[4]),
                }
            )
        except ValueError:
            continue


def sample_thermal(samples: list[dict], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True)
        samples.append(
            {
                "timestamp": time.time(),
                "text": out.stdout.strip(),
            }
        )
        stop_event.wait(1.0)


def summarize_monitoring(top_samples: list[dict], thermal_samples: list[dict]) -> dict:
    power_values = [row["power"] for row in top_samples]
    cpu_values = [row["cpu_pct"] for row in top_samples]
    thermal_warning_count = sum(
        1 for row in thermal_samples
        if "warning" in row["text"].lower() and "no thermal warning level has been recorded" not in row["text"].lower()
    )
    return {
        "top_sample_count": len(top_samples),
        "mean_power_impact": (sum(power_values) / len(power_values)) if power_values else 0.0,
        "peak_power_impact": max(power_values) if power_values else 0.0,
        "mean_cpu_pct": (sum(cpu_values) / len(cpu_values)) if cpu_values else 0.0,
        "thermal_warning_samples": thermal_warning_count,
        "thermal_sample_count": len(thermal_samples),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    binary = build_binary()
    ensure_profile(binary)

    workload_paths = sorted(WORKLOAD_DIR.glob("*.json"))
    register_rows = []

    for workload_path in workload_paths:
        workload = json.loads(workload_path.read_text(encoding="utf-8"))
        for policy in POLICIES:
            run_id = f"m4-{policy}-{workload['workload_id']}"
            output_path = RAW_DIR / f"{run_id}.json"
            top_samples: list[dict] = []
            thermal_samples: list[dict] = []
            stop_event = threading.Event()
            env = {**os.environ, "VECLIB_MAXIMUM_THREADS": "1"}
            command = [
                str(binary),
                "--mode", "run",
                "--policy", policy,
                "--workload", str(workload_path),
                "--profile", str(PROFILE_PATH),
                "--cpu-workers", "8",
                "--output", str(output_path),
            ]
            benchmark = subprocess.Popen(command, cwd=ROOT, env=env)
            top_proc = subprocess.Popen(
                ["top", "-pid", str(benchmark.pid), "-l", "0", "-s", "1", "-stats", "pid,command,cpu,mem,power"],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            top_thread = threading.Thread(target=read_top_stream, args=(top_proc, top_samples, stop_event), daemon=True)
            thermal_thread = threading.Thread(target=sample_thermal, args=(thermal_samples, stop_event), daemon=True)
            top_thread.start()
            thermal_thread.start()
            started = time.time()
            benchmark.wait()
            ended = time.time()
            stop_event.set()
            try:
                top_proc.terminate()
            except ProcessLookupError:
                pass
            top_thread.join(timeout=2)
            thermal_thread.join(timeout=2)
            if top_proc.poll() is None:
                top_proc.kill()
            if benchmark.returncode != 0:
                raise RuntimeError(f"Run failed: {run_id}")
            monitoring = summarize_monitoring(top_samples, thermal_samples)
            monitor_path = MONITOR_DIR / f"{run_id}_monitor.json"
            monitor_path.write_text(
                json.dumps({"top_samples": top_samples, "thermal_samples": thermal_samples, "summary": monitoring}, indent=2),
                encoding="utf-8",
            )
            register_rows.append(
                {
                    "run_id": run_id,
                    "policy": policy,
                    "workload_id": workload["workload_id"],
                    "family": workload["family"],
                    "seed": workload["seed"],
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
                    "duration_s": round(ended - started, 3),
                    "workload_path": str(workload_path.relative_to(ROOT)),
                    "profile_path": str(PROFILE_PATH.relative_to(ROOT)),
                    "result_path": str(output_path.relative_to(ROOT)),
                    "monitor_path": str(monitor_path.relative_to(ROOT)),
                    "mean_power_impact": round(monitoring["mean_power_impact"], 4),
                    "peak_power_impact": round(monitoring["peak_power_impact"], 4),
                    "mean_cpu_pct": round(monitoring["mean_cpu_pct"], 4),
                    "thermal_warning_samples": monitoring["thermal_warning_samples"],
                }
            )
            print(f"Completed {run_id}")

    with RUN_REGISTER.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(register_rows[0].keys()))
        writer.writeheader()
        writer.writerows(register_rows)
    print(f"Wrote {RUN_REGISTER}")


if __name__ == "__main__":
    main()
