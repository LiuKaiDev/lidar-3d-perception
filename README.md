# LiDAR 3D Perception

An engineering and research workspace for reproducible LiDAR 3D detection on
nuScenes and KITTI. It owns dataset adapters, geometry, prediction schemas,
cache provenance, evaluation, fusion, validation CLIs, reports, and tests;
OpenPCDet remains the pinned third-party detector backend.

## Current state

Phase 6 is closed and Phase 7A reproducibility entrypoints are complete.
The Phase 7B package is complete locally for GitHub review; its workflow still
needs a real remote run before CI can be called passing. **VoxelNeXt is the
default detector** because it has the strongest official mini-val mAP/NDS and
precision in the frozen comparison. CenterPoint is the baseline. E3 is retained
as a directional, sequential late-fusion ablation: it recovers complementary
far/sparse objects, at the cost of false positives and lower precision, so it is
not the default.

All Phase 6 numbers are from a previously exposed `nuScenes v1.0-mini`
exploratory protocol. `mini_val` contains two scenes and is not evidence of
full-nuScenes generalization or a SOTA claim.

## Capabilities

- KITTI and nuScenes adapters with explicit coordinate-frame transforms.
- `Box3D` and `PredictionBatch` schemas with velocity and provenance fields.
- Distance-aware, point-density-aware, matching, bootstrap, and official
  nuScenes evaluation.
- Deterministic prediction-cache validation and frozen E3 prediction-only
  fusion.
- Read-only environment/asset validators and a single-sample demo CLI.
- Machine-readable experiment manifests and report generation.

## Architecture

See [`docs/13_system_architecture.md`](docs/13_system_architecture.md) for the
component map and data flow. The short version is: dataset adapter -> detector
backend -> project schema -> optional cache/fusion -> evaluation/reporting.
Ground truth is loaded only by evaluation and analysis; detector inference does
not receive GT.

## Quick start

Use Python 3.12 and a virtual environment. The CPU checks need only the small
schema/algorithm dependencies; GPU inference additionally needs CUDA Torch,
spconv, nuScenes assets, and the pinned OpenPCDet submodule.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[ci]'
.venv/bin/python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

PYTHONPATH=. .venv/bin/python tools/validate_environment.py --profile cpu
PYTHONPATH=. .venv/bin/python tools/validate_environment.py --profile gpu
PYTHONPATH=. .venv/bin/python tools/validate_assets.py --detector voxelnext
PYTHONPATH=. .venv/bin/python tools/demo_nuscenes.py --sample-token <token>
```

The first command works without a dataset, checkpoint, cache, or GPU. Asset
validation is read-only. The demo needs a local nuScenes mini root and the
selected checkpoint; it writes one JSON `PredictionBatch` payload under
`outputs/demo/` (ignored by Git).

For the full entrypoint contract, including environment profiles, asset
precedence, cache checks, and timing scopes, see
[`docs/12_phase7_reproducibility_entrypoints.md`](docs/12_phase7_reproducibility_entrypoints.md).

Select modes explicitly:

```bash
PYTHONPATH=. .venv/bin/python tools/demo_nuscenes.py --detector centerpoint --sample-token <token>
PYTHONPATH=. .venv/bin/python tools/demo_nuscenes.py --detector e3 --sample-token <token>
```

Set `NUSCENES_ROOT` or pass `--dataset-root <path>` when the dataset is not at
the configured default. `--checkpoint <path>` is available for single-detector
modes; E3 always reads its frozen detector and fusion configs.

## Phase 6 results

The generated summary is [`reports/phase6_summary.md`](reports/phase6_summary.md).
Its numbers come directly from the committed E3/E4 JSON artifacts:

```bash
PYTHONPATH=. .venv/bin/python tools/generate_phase6_summary.py --check
```

The report links the selected distance, density, official-metric, and
complementarity figures. It distinguishes historical Phase 5 detector E2E
measurements from E3's estimated sequential total and from Phase 7A CLI wall
time.

Interpret custom recall as a diagnostic slice under the frozen 2 m matcher, not
as an official nuScenes metric. mAP/NDS remain the primary model-selection
evidence; FP and precision explain why E3's recall gain does not make it the
default.

## Tests and CI

Run the complete local suite with `PYTHONPATH=. .venv/bin/python -m pytest -q`.
GitHub Actions runs a focused CPU target on Python 3.12; it never initializes a
detector, requires CUDA, downloads nuScenes/checkpoints, or runs inference. The
workflow also runs environment validation, report `--check`, YAML/JSON parsing,
and repository boundary checks. A CI badge is intentionally omitted until the
workflow has a real successful run on GitHub.

## Data, models, and third-party code

Datasets, checkpoints, prediction caches, and runtime outputs are external or
ignored assets and never enter Git. OpenPCDet is a submodule at revision
`233f849829b6ac19afb8af8837a0246890908755`; its Apache-2.0 license is retained
in the submodule. Checkpoint URLs and SHA-256 identities are recorded in the
detector configs and [`docs/15_third_party_and_assets.md`](docs/15_third_party_and_assets.md).
The project root has no selected license yet; choosing one is a release blocker
for the owner, not a reason to block documentation or CI work.

## Limitations and roadmap

The mini exploratory protocol is small, previously exposed, and not a full
benchmark. Runtime values have explicit scopes and should not be compared across
scopes. The next release-candidate work should decide a project license, repeat
the protocol on an unseen/full split, verify a clean GitHub workflow run, and
only then consider a release/tag. Tracking and a Phase 7 tag are deliberately
out of scope here.

For an interview-style explanation, see
[`docs/14_portfolio_walkthrough.md`](docs/14_portfolio_walkthrough.md).
