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
