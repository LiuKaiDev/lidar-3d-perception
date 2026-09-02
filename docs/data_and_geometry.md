# 数据与几何约定

## 点云和坐标系

项目内部 LiDAR 坐标统一为 `x` 向前、`y` 向左、`z` 向上。KITTI Velodyne
文件是 `[x, y, z, intensity]` 的 float32 记录；nuScenes 多 sweep 点云
还保留 `time_lag`。所有长度单位为米。

`Box3D` 使用几何中心、`size=[length,width,height]` 和绕 `+z` 的右手系
`yaw`。正 yaw 将局部长度方向 `+x` 转向 `+y`。点在 box 边界内按
`1e-6 m` 容差计入。

## KITTI 标定

KITTI 链路为 `Tr_velo_to_cam` -> `R0_rect` -> `P2`。标签中的相机坐标是
底部中心和 `[height,width,length]` 尺寸；适配器先平移到几何中心，再用
组合变换的逆映射到 LiDAR 坐标，并从变换后的长度方向计算 yaw。

## nuScenes sweep

适配器依据 `sample`、`sample_data`、`calibrated_sensor` 和 `ego_pose` 的
token 链接，将历史 `LIDAR_TOP` sweep 变换到参考传感器系后再拼接。GT
框和 `sample_annotation` 只供评估及点数统计使用。更多实现细节见
`lidar_perception/datasets/nuscenes_adapter.py` 与 geometry 单元测试。

## 表示方式

PointPillars 将 XY 平面离散为 pillar 并生成 BEV pseudo-image；VoxelNeXt
使用稀疏 3D voxel。两者均由 OpenPCDet 提供网络实现，项目只负责适配、
统一 schema 和评估边界。
