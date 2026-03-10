# CAPHS Minimal Reproducibility Repository

This repository is a minimal, upload-ready reproducibility package for `CAPHS` (`Constraint-Aware Predictive Heterogeneous Scheduler`).

It contains only the code, configuration, workload data, and compact reference result tables needed to reproduce the core simulation study and the Apple M4 Pro hardware validation. It does **not** include the manuscript sources, paper figures, PDFs, bibliography files, or large intermediate artifacts.

## License

This repository is distributed under the `MIT` License. See `LICENSE`.

## Repository layout

- `Plan 6/legacy/reproducibility_package/`
  - Python simulation package
  - experiment runners for the audited main and holdout suites
  - workload JSON files for the main and holdout suites
  - minimal tests
- `Plan 6/m4_hardware_validation/`
  - Apple M4 Pro hardware validation scripts
  - Swift benchmark source for the unified-memory CPU/GPU harness
  - workload JSON files for the hardware campaign
- `Plan 6/reference_results/`
  - compact CSV references for the main claims
  - no raw traces, paper figures, or manuscript assets

## Important naming note

Some implementation identifiers still use internal provenance labels such as `plan6`, `plan6_v2`, and related variant names. These are preserved for traceability and compatibility with the original experiment scripts. In the paper, the final method is referred to as `CAPHS`.

## Quick start

### 1. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e "Plan 6/legacy/reproducibility_package"
```

### 2. Run the audited simulation campaign

```bash
python "Plan 6/legacy/reproducibility_package/experiments/run_plan6_v2_campaign.py"
```

Generated outputs will be written to:

- `Plan 6/results/v2_main_suite/`
- `Plan 6/results/v2_holdout_suite/`
- `Plan 6/results/plan6_v2_combined_pairwise.csv`
- `Plan 6/logs/`

These output directories are intentionally ignored by git.

### 3. Run the Apple M4 Pro hardware validation

Requirements:

- macOS on Apple silicon
- Xcode command line tools
- Swift toolchain with `swift build`
- access to `top` and `pmset`

Commands:

```bash
python "Plan 6/m4_hardware_validation/scripts/generate_m4_workloads.py"
python "Plan 6/m4_hardware_validation/scripts/run_m4_campaign.py"
python "Plan 6/m4_hardware_validation/scripts/analyze_m4_campaign.py"
```

Generated hardware outputs will be written to:

- `Plan 6/m4_hardware_validation/results/`

This directory is also ignored by git.

## Reference results

Compact reference CSVs are stored in:

- `Plan 6/reference_results/simulation/`
- `Plan 6/reference_results/hardware/`

These files are intended for quick comparison and review. They are not a substitute for rerunning the experiments.

## What is intentionally excluded

- manuscript `.tex` files
- paper PDFs
- generated figures and tables
- bibliography and reference PDFs
- logs, caches, and temporary files
- large raw run outputs and monitoring traces
