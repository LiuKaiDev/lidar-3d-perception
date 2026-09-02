# Third-Party Code and Assets

## Dependencies and ownership

- **OpenPCDet** is a Git submodule pinned to revision
  `233f849829b6ac19afb8af8837a0246890908755`. Its checked-in
  [LICENSE](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/LICENSE)
  is Apache License 2.0; the submodule supplies CenterPoint, VoxelNeXt, model
  configs, dataset hooks, and CUDA operators. The project does not modify it.
- **PyTorch/Torch CUDA**, **spconv**, and the **nuScenes devkit** are runtime
  dependencies. Exact versions observed in the frozen Phase 6 environment are
  Torch 2.5.1+cu124, CUDA 12.4, spconv 2.3.8, and nuscenes-devkit 1.2.0.
  CPU CI installs CPU Torch and does not import detector backends.
- KITTI and nuScenes remain governed by their respective dataset terms. This
  repository stores adapters and metadata, not raw data.

## Model assets

The checkpoint sources were verified against the model-zoo table in the pinned
[OpenPCDet README](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/README.md).
The detector YAML
files preserve those source URLs and the locally validated SHA-256 identities:

| Model | Config | Checkpoint SHA-256 |
| --- | --- | --- |
| CenterPoint-PointPillar | `configs/detectors/centerpoint/nuscenes_mini.yaml` | `955a3e38868b81f6ae74f09f84a774ef002d03484c6a8e1194b147069c0a6c2a` |
| VoxelNeXt | `configs/detectors/voxelnext/nuscenes_mini.yaml` | `9409dd8c13c8c8ca546c8c5af024856d03029e2def8d5fc0fa3bfe4477e7d88b` |

Checkpoints require a local path and are ignored by Git. Datasets, prediction
caches, evaluator dumps, and runtime outputs are also ignored; validators fail
or warn when required assets are absent rather than downloading them.

## Project license status

There is currently no root-level `LICENSE`. The project-level license is left for
the owner to decide and is a Phase 7C release blocker. This is intentional: no
license terms are inferred from OpenPCDet or dataset terms.

The project-owned scope is the adapter, geometry, schema, cache/provenance,
matching/evaluation, frozen prediction-only fusion, validation/demo tooling,
reports, and tests around those boundaries.
