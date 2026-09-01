# Third-party dependencies

`OpenPCDet` is tracked as a Git submodule at the frozen Phase 6 revision
`233f849829b6ac19afb8af8837a0246890908755`. Its source, configs, and CUDA ops
remain third-party and are not copied into `lidar_perception/`; the project
adapter is the only integration boundary. See
[`docs/15_third_party_and_assets.md`](../docs/15_third_party_and_assets.md) for
asset identities, licensing evidence, and redistribution boundaries.

CPU tests do not import OpenPCDet or initialize detector models. GPU validation
and the demo require a checked-out submodule, compatible CUDA Torch/spconv, the
nuScenes devkit, a local dataset, and a verified checkpoint.
