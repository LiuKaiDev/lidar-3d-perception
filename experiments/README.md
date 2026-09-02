# 实验索引

Phase 6 是在 `nuScenes v1.0-mini` 上进行的受计算资源约束的 exploratory
experiment。以下目录中的 `config.yaml`、`metrics.json`、`benchmark.json`、
`analysis.md`、环境记录和 figures 是原始依据，不修改其内容来配合新的文档。

| ID | 实验 | 状态 | 参数选择 | 确认方式 |
|---|---|---|---|---|
| E0 | 冻结 baseline protocol | PASS | 已验证记录 | `mini_val` |
| E1 | 预测距离校准 | NEGATIVE | 仅 `mini_train` | 冻结设置上的 `mini_val` |
| E2 | 预测框稀疏度策略 | DIRECTIONAL | 仅 `mini_train` | 冻结设置上的 `mini_val` |
| E3 | CenterPoint + VoxelNeXt late fusion | DIRECTIONAL | 仅 `mini_train` | 冻结设置上的 `mini_val` |
| E4 | 重复验证和最终 ablation | PASS | 不在 `mini_val` 调参 | 冻结 cache 重复评估 |

`mini_val` 只有两个 scene，且此前已暴露；结果不代表 full nuScenes
benchmark 或 SOTA。PredictionCache、logs、checkpoint 和运行输出均被 Git
忽略。实验目录模板位于 `_template/`。
