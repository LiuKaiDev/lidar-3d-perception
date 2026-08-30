# Coordinate Systems and Conventions

## Internal Convention

All project-owned Phase 1 geometry uses the KITTI Velodyne/LiDAR frame:

```text
x: forward
y: left
z: up
```

`Box3D` stores:

```text
center = geometric center [x, y, z]
size   = [length, width, height]
yaw    = rotation around +z, positive by the right-hand rule
```

With this convention a positive yaw rotates the local +x (length) direction
towards +y. Points on a box face are inside under the inclusive `1e-6` meter
boundary tolerance used by `point_in_box`.

## KITTI Calibration

KITTI Object Detection calibration provides the following mappings:

```text
Tr_velo_to_cam: Velodyne -> unrectified camera
R0_rect:        camera -> rectified camera
P2:             rectified camera -> image homogeneous coordinates
```

For a Velodyne point `p_velo` represented homogeneously:

```text
p_rect = R0_rect * Tr_velo_to_cam * p_velo
q      = P2 * p_rect
pixel  = [q_x / q_z, q_y / q_z]
depth  = p_rect_z
```

`KittiCalibration` stores the named matrices and exposes both forward and
inverse transforms. The inverse is computed from the composed homogeneous
matrix, so the direction is explicit rather than inferred from a generic
`matrix2` variable.

## Camera and LiDAR Axes

The rectified camera frame uses `x` right, `y` down and `z` forward. A KITTI
label row stores its 3D location at the **bottom center** in this camera frame,
dimensions in native `[height, width, length]` order, and `rotation_y` around
camera +y. The adapter converts it as follows:

```text
camera bottom center
    + [0, -height/2, 0]
    -> camera geometric center
    -> inverse(R0_rect * Tr_velo_to_cam)
    -> internal LiDAR center
```

The camera-frame local length direction is
`[cos(rotation_y), 0, -sin(rotation_y)]`. It is transformed as a direction
(without translation) into LiDAR coordinates, then its XY angle becomes the
internal yaw. This derives the heading from the actual calibration instead of
hard-coding a dimension swap or a sign-only formula.

## Projection Validity

`project_points_to_image` returns pixel coordinates, rectified-camera depth,
and a validity mask. Points with `depth <= 1e-6` are invalid and receive NaN
pixels; valid points remain finite even if they lie outside the image. Box
projection applies the same rule independently to each of eight corners, so a
box partly behind the camera produces a finite partial projection rather than
contaminating every corner.

## Transformation Composition

The transform API uses:

```text
p_destination = T_destination_source * p_source
compose_transform(T_ab, T_bc) = T_ab * T_bc
```

This convention is tested with identity, translation, 90-degree rotation,
composition, inversion and quaternion round trips in the Phase 1 unit tests.

## Real KITTI Validation

The convention and calibration chain were validated on real KITTI training
frames `000000`, `004139`, and `007480`. For each frame, the project-owned
adapter loaded the Velodyne cloud, parsed `P2`, `R0_rect`, and
`Tr_velo_to_cam`, converted every labeled object to an internal `Box3D`, and
computed oriented point counts. LiDAR -> rectified-camera -> LiDAR round-trip
errors were at most `1.42e-14` m on sampled real points.

Saved visual checks are:

```text
outputs/phase1_validation/000000_bev.png
outputs/phase1_validation/004139_bev.png
outputs/phase1_validation/007480_bev.png
outputs/phase1_validation/004139_3d.png
outputs/phase1_validation/000000_image.png
outputs/phase1_validation/004139_image.png
outputs/phase1_validation/007480_image.png
outputs/phase1_validation/kitti_training_stats.json
```

The BEV views show the converted oriented boxes in the corresponding point
clouds. The image views show LiDAR projections and 3D box edges aligned with
the camera scenes; boxes at the image boundary are naturally clipped by the
camera field of view. The 3D check uses the non-interactive matplotlib output
path because it is deterministic in WSL; Open3D 0.19.0 is also installed for
interactive use.

Representative native-to-internal conversions were:

| Frame/class | Native location | Native h/w/l | rotation_y | Internal center | Internal l/w/h | yaw | points |
|---|---|---|---:|---|---|---:|---:|
| 000000 Pedestrian | [1.84, 1.47, 8.41] | [1.89, 0.48, 1.20] | 0.010 | [8.74, -1.87, -0.65] | [1.20, 0.48, 1.89] | -1.582 | 377 |
| 004139 Pedestrian | [-9.94, 1.08, 23.30] | [1.82, 0.78, 0.94] | -1.770 | [23.65, 9.88, -0.23] | [0.94, 0.78, 1.82] | 0.198 | 66 |
| 004139 Car | [13.75, 1.51, 23.97] | [1.48, 1.56, 3.33] | -1.640 | [24.28, -13.80, -1.14] | [3.33, 1.56, 1.48] | 0.068 | 85 |
| 004139 Cyclist | [-5.71, 1.13, 31.15] | [1.79, 0.51, 1.73] | 1.540 | [31.49, 5.64, -0.39] | [1.73, 0.51, 1.79] | -3.112 | 40 |
| 007480 Car | [-6.96, 1.73, 7.83] | [1.56, 1.57, 4.37] | -3.130 | [8.11, 6.97, -0.87] | [4.37, 1.57, 1.56] | 1.559 | 658 |

For five sampled boxes, OpenPCDet's fixed KITTI conversion was used only as an
optional third-party cross-check. Size differences were below `6e-8`, center
differences were about `0.012 m` (float32 reference versus float64 project
calculation), and yaw differences were below `0.002 rad`. OpenPCDet source was
not modified or copied.
