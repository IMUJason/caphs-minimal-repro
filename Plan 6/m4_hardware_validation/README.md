# Apple M4 Pro Hardware Validation

This directory contains the minimal code and workload data needed to reproduce the real-device CAPHS validation on Apple M4 Pro.

Included:

- Swift benchmark source (`swift_bench/`)
- workload generator
- hardware campaign runner
- analysis script
- workload JSON files

Excluded:

- generated paper figures
- manuscript tables
- large raw monitoring artifacts

Typical workflow:

```bash
python "Plan 6/m4_hardware_validation/scripts/generate_m4_workloads.py"
python "Plan 6/m4_hardware_validation/scripts/run_m4_campaign.py"
python "Plan 6/m4_hardware_validation/scripts/analyze_m4_campaign.py"
```

Notes:

- This workflow is macOS-specific.
- It expects Apple silicon with CPU/GPU unified memory.
- The power metric used by the campaign is the macOS `top` power-impact field, not board-level watts.

