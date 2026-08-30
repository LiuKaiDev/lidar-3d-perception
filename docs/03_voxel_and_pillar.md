# Voxel and Pillar Representation

Raw LiDAR points are an irregular set, while a conventional 2D CNN expects a
regular grid. PointPillars discretizes the XY ground plane into vertical
columns, called pillars, and aggregates the points in each column into a fixed
feature vector.

The fixed OpenPCDet KITTI config uses:

```text
point cloud range: [0, -39.68, -3, 69.12, 39.68, 1]
pillar size:       [0.16, 0.16, 4.0] meters
max points/pillar: 32
max pillars:       16000 train / 40000 test
```

The Z size spans the configured vertical range, so there is no Z subdivision
in the pillar grid. This is cheaper than a full 3D voxel grid and produces a
2D pseudo-image for the backbone. Smaller pillars preserve more spatial
detail but increase active pillars, memory and latency. Larger pillars reduce
cost while losing fine geometry. The max-points and max-pillars limits bound
memory and make batch processing predictable; truncation can discard detail in
dense regions.

PointPillars is distinct from a sparse 3D convolution pipeline: its learned
per-pillar features are scattered directly into a BEV pseudo-image, and the
2D backbone does the spatial reasoning.

For this project, OpenPCDet's `data/kitti/training` and `testing` directories
are symbolic links to `~/datasets/kitti`; the 39 GB dataset is not copied into
the repository. The official preparation command generated the ignored
`kitti_infos_{train,val,trainval,test}.pkl` files and GT database required by
the detector data pipeline.
