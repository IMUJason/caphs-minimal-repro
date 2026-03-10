# CAPHS Simulation Reproducibility Package

This package contains the minimal Python code required to reproduce the audited CAPHS simulation campaign.

Included:

- simulator backend
- policy implementations
- workload generation logic
- campaign runners for the main and holdout suites
- compact tests

Excluded:

- manuscript assets
- paper figure generators
- large raw result directories

The canonical command is:

```bash
python "Plan 6/legacy/reproducibility_package/experiments/run_plan6_v2_campaign.py"
```

The generated outputs are written under `Plan 6/results/` and `Plan 6/logs/`, which are ignored by git in this repository.

