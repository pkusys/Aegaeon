# 安装

Aegaeon 支持 Linux x86_64 和 aarch64，要求标准 CPython 3.10。x86_64 可使用 CUDA 12.9 或 13.0；aarch64（包括 DGX Spark）使用 CUDA 13.0。

## 预编译 wheel

wheel 包含 Aegaeon 的 CUDA 和 Rust 扩展。运行时内核编译仍需匹配的 CUDA Toolkit、C++ 编译器和 Ninja，工具链命令见[源码安装](#源码安装)。Rust/Cargo 仅用于源码构建。每种 CUDA 路线使用独立的 Python 3.10 环境。

从 [GitHub Releases](https://github.com/pkusys/aegaeon/releases) 下载匹配 CUDA 版本、CPU 架构和 CPython 3.10 的 wheel。激活 Python 环境，在仅包含这一份 Aegaeon wheel 的目录中执行安装命令。

### CUDA 12.9：Linux x86_64

预编译发行版支持 Linux x86_64、标准 CPython 3.10，以及 glibc >=2.31。

```bash
python -m pip install ./aegaeon-*.whl \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://wheels.vllm.ai/0.26.0/cu129
```

### CUDA 13.0：DGX Spark / Linux aarch64

DGX Spark 预编译 wheel 支持 Linux aarch64、标准 CPython 3.10 和 glibc >=2.31，需要兼容 CUDA 13.0 的 NVIDIA 驱动。

在 DGX Spark 的 Linux 终端安装 CUDA 13.0 aarch64 wheel：

```bash
python -m pip install ./aegaeon-*.whl
```

## 源码安装

源码构建要求 glibc >=2.28、匹配的 CUDA Toolkit（包含 `nvcc`）、C++ 编译器、Ninja 和 Rust/Cargo。在仓库根目录执行：

```bash
conda create --solver classic -n aegaeon python=3.10 -y
conda activate aegaeon
```

| 选择 | CUDA Toolkit | PyTorch | vLLM | 平台 |
|---|---|---|---|---|
| `cu129` | 12.9 | 2.11.0+cu129 | 0.26.0+cu129 | Linux x86_64 |
| `cu130` | 13.0 | 2.11.0 | 0.26.0 | Linux x86_64 / aarch64 |

需要兼容的 NVIDIA 驱动，见 NVIDIA 的 [CUDA 兼容性指南](https://docs.nvidia.com/deploy/cuda-compatibility/)。

### CUDA 12.9

该路线支持 Linux x86_64。如果尚未准备匹配的工具链，可以在 Conda 环境中手动安装：

```bash
conda install --solver classic -c nvidia cuda-toolkit=12.9 \
  gcc_linux-64=11.2.0 gxx_linux-64=11.2.0 rust ninja -y
conda activate aegaeon
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

如果使用系统 Toolkit，将 `CUDA_HOME` 设为其根目录。上述 Conda 示例使用 GCC/G++ 11.2。

安装匹配的依赖，然后编译安装 Aegaeon：

```bash
python -m pip install -r requirements-cu129.txt
AEGAEON_CUDA=cu129 python -m pip install --no-build-isolation -e .
```

### CUDA 13.0

Linux x86_64：

```bash
conda install --solver classic -c nvidia cuda-toolkit=13.0.2 \
  gcc_linux-64=11.2.0 gxx_linux-64=11.2.0 rust ninja -y
conda activate aegaeon
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

Linux aarch64 / DGX Spark 使用系统 CUDA 13.0 Toolkit 和 GCC/G++ 13：

```bash
conda install --solver classic -c conda-forge rust ninja -y
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

然后安装 CUDA 13.0 依赖并编译 Aegaeon：

```bash
python -m pip install -r requirements-cu130.txt
AEGAEON_CUDA=cu130 python -m pip install --no-build-isolation -e .
```

## 环境检查

```bash
python -m pip check
aegaeon doctor
aegaeon doctor --check-cuda
```

在 Linux aarch64 / CUDA 13.0 环境中，`pip check` 可能报 `nvidia-cusparselt-cu13 0.8.0 is not supported on this platform`。这是 NVIDIA 上游 wheel 的元数据问题：文件名中的平台标签为 `manylinux2014_aarch64`，内部却声明为 `manylinux2014_sbsa`，导致兼容性检查失败。

Foundry 图 SAVE/LOAD 为可选功能，需要另行安装 Foundry，见 [CUDA Graph](cuda-graphs.md)。接下来阅读[快速开始](quickstart.md)。安装错误见[故障排查](troubleshooting.md)。
