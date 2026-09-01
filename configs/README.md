# Configs

Configuration is split by ownership and use:

- `detectors/`: frozen third-party detector wrappers and checkpoint identities.
- `analysis/` and `benchmark/`: validated Phase 4/5 protocols.
- `system/portfolio.yaml`: Phase 7 engineering entrypoint selection. It defaults
  to VoxelNeXt and references existing detector/E3 configs rather than copying
  checkpoint hashes or fusion parameters.

Historical E0-E4 experiment manifests and frozen parameters live under
`experiments/` and must not be changed by Phase 7 entrypoints.
