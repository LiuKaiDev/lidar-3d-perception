# Phase 0 Environment Notes

Status: **Validated / PASS**

This document explains how the validated environment was assembled. The exact
machine values and package versions are frozen in
[`environment.lock.md`](environment.lock.md).

## Architecture

The machine uses Windows 10 build `10.0.19045.6466` with WSL2 and Ubuntu
`24.04.4 LTS`. The NVIDIA display driver stays on the Windows host. WSL sees
the host driver through GPU paravirtualization; no Linux NVIDIA display driver
was installed in Ubuntu.

`nvidia-smi` reports the driver's supported CUDA capability (`13.1`). This is
not the same value as the CUDA Toolkit used by `nvcc` or the CUDA version
against which PyTorch was built.

## Validated Combination

The project uses a local `.venv` because Conda/Mamba is not installed. The
validated Python stack is:

```text
Python 3.12.3
PyTorch 2.5.1+cu124 (torch.version.cuda = 12.4)
torchvision 0.20.1+cu124
spconv-cu124 2.3.8
OpenPCDet 233f849829b6ac19afb8af8837a0246890908755
```

OpenPCDet's official setup declares CUDA extensions in `setup.py`; it was
installed in editable mode with `--no-build-isolation` so the build could use
the already validated Torch installation. The source tree itself was not
modified.

The host Ubuntu image has GCC 13.3, while the Debian CUDA 12.0 compiler in
this WSL image does not accept GCC 13. A user-space GCC 12.4 toolchain was
therefore unpacked under the ignored `.gcc-12/` directory and selected only for
the OpenPCDet build. The Toolkit is likewise a user-space extraction under
the ignored `.cuda-toolkit-12.0/` directory; it does not install or replace a
system driver.

`nvcc` is 12.0.140 and PyTorch is compiled with CUDA 12.4. PyTorch emits a
minor-version warning during extension builds, but all OpenPCDet extensions
compiled and the GPU operation smoke tests passed. Keep these versions fixed;
do not randomly upgrade CUDA or Torch when reproducing this environment.

## Reproduction Outline

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e . --no-deps --no-build-isolation
```

Install the exact packages and hashes listed in `environment.lock.md`, then
add OpenPCDet as the fixed submodule:

```bash
git submodule update --init --depth 1 third_party/OpenPCDet
git -C third_party/OpenPCDet checkout 233f849829b6ac19afb8af8837a0246890908755
```

Build with the validated user-space compiler view. The command below is the
actual build command used in Phase 0; it intentionally limits compilation to
the RTX 2060's `sm_75` architecture and one job:

```bash
CUDA_HOME="$PWD/.cuda-home-12.0" \
PATH="$PWD/.cuda-home-12.0/bin:$PWD/.gcc-12/usr/bin:$PATH" \
CC="$PWD/.gcc-12/usr/bin/gcc-12" \
CXX="$PWD/.gcc-12/usr/bin/g++-12" \
TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=1 \
NVCC_PREPEND_FLAGS="-I$PWD/.cuda-home-12.0/include" \
python -m pip install -e third_party/OpenPCDet --no-build-isolation --no-deps
```

The complete package-install and user-space Toolkit extraction commands are
recorded in the lock document rather than hidden in a project script.

## Smoke Tests

Run the commands in the `Smoke Tests` section of `environment.lock.md`. The
important distinction is that `import pcdet` alone is insufficient: the
validated test executes GPU BEV IoU/NMS and `roiaware_pool3d` point-in-box
operations from the compiled OpenPCDet extensions.

## Known Limitations

- No dataset, checkpoint, training run, or detector integration was added in
  Phase 0.
- `nvcc`/Toolkit is a local user-space build tool because root access was not
  available; it is not a system-wide CUDA installation.
- The minor Toolkit/PyTorch CUDA mismatch is observed and tested, but should
  remain frozen for reproducibility.
