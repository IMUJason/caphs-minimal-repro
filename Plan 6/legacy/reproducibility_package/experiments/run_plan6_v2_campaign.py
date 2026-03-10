from __future__ import annotations

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from plan6_scheduler.experiment import run_experiment_suite

from analyze_plan6_v2_suite import analyze_suite
from summarize_plan6_v2_campaign import main as summarize_campaign


def main() -> None:
    config_dir = Path(__file__).resolve().parent / "configs"
    config_paths = [
        config_dir / "plan6_v2_main_suite.yaml",
        config_dir / "plan6_v2_holdout_suite.yaml",
    ]
    for config_path in config_paths:
        outputs = run_experiment_suite(config_path)
        print(f"completed_experiment: {config_path}")
        for key, value in outputs.items():
            print(f"{key}: {value}")
        analyze_outputs = analyze_suite(outputs["summary_metrics"].parent, focus_policy="plan6_v2")
        for key, value in analyze_outputs.items():
            print(f"{key}: {value}")
    summarize_campaign()


if __name__ == "__main__":
    main()
