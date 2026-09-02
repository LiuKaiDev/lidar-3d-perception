# 配置

- `detectors/`：CenterPoint、VoxelNeXt 和 PointPillars 的 OpenPCDet 适配配置。
- `analysis/`、`benchmark/`：历史评估和基准协议。
- `system/portfolio.yaml`：默认 detector、数据路径和 E3 冻结配置入口。

模型 checkpoint、数据集和运行时 cache 不在仓库中。配置中的来源 URL、
SHA-256、split 和阈值是复现实验边界，不应在没有新实验协议的情况下修改。
