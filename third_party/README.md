# Third-party Dependencies

`OpenPCDet` will live under this directory during Phase 0.

Target structure:

```text
third_party/
└── OpenPCDet/
```

Do not copy OpenPCDet source files into the project's own Python package.

During Phase 0, Codex should:

1. inspect the actual WSL / GPU / Python / PyTorch environment;
2. determine a compatible OpenPCDet revision and dependency combination;
3. prefer a fixed Git submodule / fixed commit strategy;
4. compile and smoke-test required CUDA ops;
5. record the exact commit and environment in `docs/environment.lock.md`.

No OpenPCDet source modification should be made during repository bootstrap.
