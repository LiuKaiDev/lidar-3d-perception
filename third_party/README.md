# 第三方依赖

`OpenPCDet` 以 Git submodule 形式固定在
`233f849829b6ac19afb8af8837a0246890908755`。其源码、模型配置和 CUDA 算子
属于第三方，项目不复制或修改；请保留子模块中的 Apache-2.0 LICENSE 和
版权声明。项目 adapter 位于 `lidar_perception/detection/`。

CPU 测试不初始化该子模块。GPU validator 和 demo 需要已 checkout 的
OpenPCDet、兼容的 CUDA Torch/spconv、nuScenes 数据和 checkpoint。
更多来源、hash 和许可证边界见 [docs/third_party.md](../docs/third_party.md)。
