# 文档导航

这里按使用场景组织项目文档。冻结实验数据和原始协议仍保留在
`experiments/` 与 `docs/archive/`，不在功能文档中重复维护。

| 文档 | 内容 |
|---|---|
| [quickstart.md](quickstart.md) | CPU 检查、GPU 资产检查和单样本推理 |
| [environment.md](environment.md) | 安装层次、CUDA/OpenPCDet 兼容环境与验证方式 |
| [architecture.md](architecture.md) | 数据流、模块职责、cache、evaluation 和 fusion 边界 |
| [data_and_geometry.md](data_and_geometry.md) | KITTI/nuScenes 数据、坐标系、Box3D 和点云变换 |
| [evaluation.md](evaluation.md) | matching、距离/密度指标、官方评估和 runtime 语义 |
| [third_party.md](third_party.md) | OpenPCDet、模型、数据集和许可证边界 |
| [environment.lock.md](environment.lock.md) | 已验证机器环境的原始快照 |
| [archive/README.md](archive/README.md) | 保留的历史协议和测量记录 |

实验结果入口是 [experiments/README.md](../experiments/README.md) 和
[reports/phase6_summary.md](../reports/phase6_summary.md)。
