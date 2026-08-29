# KITTI Point Cloud Basics

## One Frame

KITTI Velodyne files are binary float32 records with four values per point:

```text
[x, y, z, intensity]
```

The project loads them as an `N x 4` NumPy array. `x` points forward in the
Velodyne frame, `y` points left, `z` points up, and intensity is the returned
laser reflectance value. The loader checks that the byte count is divisible by
16 and rejects non-finite values rather than silently reshaping malformed data.

## Range and Sparsity

Sensor range is computed from the XYZ columns:

```text
r = sqrt(x^2 + y^2 + z^2)
```

LiDAR samples directions and surfaces, not every location in a volume. The
same angular resolution covers a larger physical area as distance increases,
so a fixed-size object receives fewer returns. Occlusion, incidence angle and
reflectivity reduce it further. This is why distance and point density are
separate, useful signals for later evaluation.

## Box Point Counts

`count_points_in_box` applies the oriented box transform in the internal LiDAR
frame and counts points with an inclusive boundary tolerance. It is a
ground-truth statistic only in Phase 1; detector matching and density-aware
evaluation belong to later phases.

## KITTI Pose Limitation

`PointCloudFrame` always exposes `lidar_to_ego` and `ego_to_global`. KITTI
Object Detection does not provide true ego/global temporal poses, so the Phase
1 adapter uses identity matrices and records this fact in frame metadata. No
fake trajectory or timestamp is introduced.
