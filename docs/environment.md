# 环境与安装

## 依赖层次

- **CPU 检查**：Python 3.12、`numpy`、`PyYAML`、`pytest` 和 CPU Torch，见
  `.[ci]`。
- **GPU 推理**：在 CPU 依赖之外增加 CUDA 版 Torch、spconv、nuScenes
  devkit、已初始化并编译的 OpenPCDet，以及本地 checkpoint 和数据集。
- **项目包**：`python -m build` 生成的 wheel 包含 Python 包、YAML 配置和
  LICENSE；`tools/` 下的验证器和 demo 是源码 checkout CLI，不声明为
  安装后的 console script。

## 验证

```bash
PYTHONPATH=. .venv/bin/python tools/validate_environment.py --profile cpu
PYTHONPATH=. .venv/bin/python tools/validate_environment.py --profile gpu
PYTHONPATH=. .venv/bin/python tools/validate_assets.py --detector voxelnext
```

CPU profile 只检查项目结构、配置和轻量依赖；GPU profile 额外检查 CUDA、
spconv、nuScenes devkit、固定 OpenPCDet revision 和编译扩展。validator
只读，不安装依赖、下载资产、运行 inference 或生成 cache。

## 已验证快照

`environment.lock.md` 是一次 RTX 2060/WSL 环境的原始记录，包含 Torch
2.5.1+cu124、CUDA 12.4、spconv 2.3.8、nuScenes devkit 1.2.0 和
OpenPCDet revision `233f849829b6ac19afb8af8837a0246890908755`。其中的本机
路径、编译命令和版本差异属于历史复现证据，不代表所有机器的默认路径。
