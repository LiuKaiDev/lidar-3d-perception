# 第三方代码与资产

## OpenPCDet

OpenPCDet 是 Git submodule，固定 revision
`233f849829b6ac19afb8af8837a0246890908755`。其网络、配置、数据 hook 和
CUDA 算子属于第三方，保留子模块自己的 Apache-2.0 LICENSE 与版权声明；
项目 adapter、schema、cache、evaluation、fusion、验证工具和测试属于本项目。

## 模型与数据

CenterPoint 和 VoxelNeXt checkpoint 来源、配置路径及 SHA-256 保存在
`configs/detectors/`：

| 模型 | SHA-256 |
|---|---|
| CenterPoint-PointPillar | `955a3e38868b81f6ae74f09f84a774ef002d03484c6a8e1194b147069c0a6c2a` |
| VoxelNeXt | `9409dd8c13c8c8ca546c8c5af024856d03029e2def8d5fc0fa3bfe4477e7d88b` |

nuScenes、KITTI、预训练 checkpoint、spconv、Torch/CUDA 和 nuScenes devkit
遵循各自来源条款。dataset、checkpoint、PredictionCache、evaluator dump
和 runtime outputs 不随项目仓库或 release 分发。

## 项目许可证

项目原创代码和文档使用根目录 [Apache-2.0 LICENSE](../LICENSE)。该许可
不重新授权 OpenPCDet、数据集、checkpoint 或其他第三方资产；它们的来源
声明和许可证仍然有效。
