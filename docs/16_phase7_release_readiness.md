# Phase 7 Release Readiness

Status: **release preparation complete; formal release blocked**.

## Acceptance Scope

- Public entrypoints identify VoxelNeXt as default, CenterPoint as baseline,
  and E3 as a directional ablation with both recall gains and accuracy/FP costs.
- The Phase 6 report is generated from frozen E3/E4 JSON artifacts and reuses
  four compact committed figures.
- The asset-free Python 3.12 CPU path validates schemas, geometry, matching,
  bootstrap, cache provenance, manifests, E3 prediction-only fusion, Phase 7
  entrypoints, documents, and deterministic report generation.
- GPU, dataset, checkpoint, cache, third-party, and runtime-output boundaries
  are explicit. No model or dataset asset is distributed.
- OpenPCDet remains pinned at
  `233f849829b6ac19afb8af8837a0246890908755` and unmodified.

Remote CI evidence is authoritative only when the `CPU tests` workflow on
`main` succeeds for the exact final commit. The run URL, head SHA, event, and
conclusion are recorded in the Phase 7C handoff and remain available in GitHub
Actions; this document intentionally does not create a self-referential commit
SHA loop.

## Release Gates

- **Required before tag/release:** project owner selects a root project license.
- **Required before tag/release:** synchronize `pyproject.toml` version `0.1.0`
  with the approved release/tag version.
- **Required before tag/release:** final review of the release-notes draft and
  successful CPU workflow for the release-candidate commit.
- **Future work, not this engineering gate:** unseen/full nuScenes validation.

No Phase 7 tag or GitHub Release has been created. See the
[v0.7.0-rc1 draft](releases/v0.7.0-rc1.md).
