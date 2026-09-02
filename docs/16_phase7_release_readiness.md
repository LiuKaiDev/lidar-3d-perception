# Phase 7 Release Readiness

Status: **release candidate prepared; not a stable release**.

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

- **Completed for this RC:** root Apache-2.0 license added and referenced in
  package metadata and user-facing documentation.
- **Completed for this RC:** `pyproject.toml` version synchronized to
  `0.7.0rc1` and the final CPU workflow succeeded for the tagged commit.
- **Completed for this RC:** release notes reviewed and published as the GitHub
  prerelease body.
- **Packaging boundary:** the wheel contains the project Python package,
  Apache-2.0 license, and YAML configs under
  `share/lidar-3d-perception/configs`. Repository tools such as environment
  validation and the demo remain source-checkout CLIs rather than installed
  console scripts; wheel smoke testing imports the installed package, exercises
  schema/config parsing, and invokes those source tools against the wheel
  environment.
- **Future work, not this engineering gate:** unseen/full nuScenes validation.
- **Future work, not this engineering gate:** tracking and PyPI publication.

The stable `v0.7.0` release remains intentionally out of scope. See the
[v0.7.0-rc1 release notes](releases/v0.7.0-rc1.md).
