# Installation

Aegaeon supports Linux x86_64 and aarch64 with standard CPython 3.10. Use CUDA 12.9 on x86_64, or CUDA 13.0 on x86_64 and aarch64, including DGX Spark.

## Pre-built wheels

Wheels include Aegaeon's CUDA and Rust extensions. Runtime kernel compilation still requires a matching CUDA Toolkit, a C++ compiler, and Ninja; see the toolchain commands under [From-source installation](#from-source-installation). Rust/Cargo is required only for source builds. Install each CUDA route in a fresh Python 3.10 environment.

Download a wheel matching your CUDA version, CPU architecture, and CPython 3.10 from [GitHub Releases](https://github.com/pkusys/aegaeon/releases). Activate the Python environment and run the installation command from a directory containing only that Aegaeon wheel.

### CUDA 12.9: Linux x86_64

The precompiled release supports Linux x86_64, standard CPython 3.10, and glibc >=2.31.

```bash
python -m pip install ./aegaeon-*.whl \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://wheels.vllm.ai/0.26.0/cu129
```

### CUDA 13.0: DGX Spark / Linux aarch64

The precompiled DGX Spark wheel supports Linux aarch64, standard CPython 3.10, and glibc >=2.31. A CUDA 13.0-compatible NVIDIA driver is required.

Install the CUDA 13.0 aarch64 wheel in a Linux terminal on DGX Spark:

```bash
python -m pip install ./aegaeon-*.whl
```

## From-source installation

Source builds require glibc >=2.28, a matching CUDA Toolkit (including `nvcc`), a C++ compiler, Ninja, and Rust/Cargo. Run from the repository root:

```bash
conda create --solver classic -n aegaeon python=3.10 -y
conda activate aegaeon
```

| Choice | CUDA Toolkit | PyTorch | vLLM | Platform |
|---|---|---|---|---|
| `cu129` | 12.9 | 2.11.0+cu129 | 0.26.0+cu129 | Linux x86_64 |
| `cu130` | 13.0 | 2.11.0 | 0.26.0 | Linux x86_64 / aarch64 |

A compatible NVIDIA driver is required; see NVIDIA's [CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/).

### CUDA 12.9

This route supports Linux x86_64. If you do not already have a matching toolchain, install it manually in the Conda environment:

```bash
conda install --solver classic -c nvidia cuda-toolkit=12.9 \
  gcc_linux-64=11.2.0 gxx_linux-64=11.2.0 rust ninja -y
conda activate aegaeon
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

With a system Toolkit, set `CUDA_HOME` to its root instead. The Conda example uses GCC/G++ 11.2.

Install the matching dependencies, then build and install Aegaeon:

```bash
python -m pip install -r requirements-cu129.txt
AEGAEON_CUDA=cu129 python -m pip install --no-build-isolation -e .
```

### CUDA 13.0

On Linux x86_64:

```bash
conda install --solver classic -c nvidia cuda-toolkit=13.0.2 \
  gcc_linux-64=11.2.0 gxx_linux-64=11.2.0 rust ninja -y
conda activate aegaeon
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

On Linux aarch64 / DGX Spark, use the system CUDA 13.0 Toolkit and GCC/G++ 13:

```bash
conda install --solver classic -c conda-forge rust ninja -y
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

Then install the CUDA 13.0 dependencies and build Aegaeon:

```bash
python -m pip install -r requirements-cu130.txt
AEGAEON_CUDA=cu130 python -m pip install --no-build-isolation -e .
```

## Check the environment

```bash
python -m pip check
aegaeon doctor
aegaeon doctor --check-cuda
```

On Linux aarch64 with CUDA 13.0, `pip check` may report `nvidia-cusparselt-cu13 0.8.0 is not supported on this platform`. This is an upstream NVIDIA wheel metadata issue: the filename uses `manylinux2014_aarch64`, but the internal tag declares `manylinux2014_sbsa`, causing the compatibility check to fail.

Foundry graph SAVE/LOAD is optional and requires a separate Foundry installation; see [CUDA graphs](cuda-graphs.md). Continue with the [Quickstart](quickstart.md). For installation errors, see [Troubleshooting](troubleshooting.md).
