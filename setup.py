import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from aegaeon_cuda import (python_abi_tag, release_mode, runtime_requirements,
                         select_backend)

python_abi_tag()

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


ROOT_DIR = Path(__file__).resolve().parent
BUILD_CUDA = os.environ.get("AEGAEON_BUILD_CUDA", "1")
if BUILD_CUDA not in {"0", "1"}:
    raise ValueError("AEGAEON_BUILD_CUDA must be 0 or 1")
SDIST = os.environ.get("AEGAEON_BUILDING_SDIST") == "1"
RELEASE = release_mode()
if RELEASE and not SDIST:
    if os.environ.get("AEGAEON_CUDA") not in {"cu129", "cu130"} or BUILD_CUDA != "1":
        raise ValueError("PyPI release wheels require explicit AEGAEON_CUDA=cu129 or cu130 "
                         "and AEGAEON_BUILD_CUDA=1.")
ROUTE = os.environ.get("AEGAEON_RESOLVED_CUDA")
if ROUTE is None:
    ROUTE = (os.environ["AEGAEON_CUDA"] if RELEASE and not SDIST else
             select_backend(os.environ.get("AEGAEON_CUDA")).route)
if ROUTE == "core" and BUILD_CUDA == "1" and not SDIST:
    raise ValueError("AEGAEON_CUDA=core requires AEGAEON_BUILD_CUDA=0")
NATIVE = BUILD_CUDA == "1" and not SDIST


def read_readme():
    path = ROOT_DIR / "README.md"
    return io.open(path, encoding="utf-8").read() if path.is_file() else ""


def get_requirements():
    # Runtime and build tools are listed separately for explicit source installs.
    common = (ROOT_DIR / "requirements-common.txt").read_text(encoding="utf-8")
    return [line.strip() for line in common.splitlines()
            if line.strip() and not line.lstrip().startswith("#")] + runtime_requirements(
                ROUTE, registry=RELEASE and not SDIST)


class BuildPy(build_py):
    """Bundle runtime profiles without shipping local tools or deployment files."""

    def find_package_modules(self, package, package_dir):
        return [
            module for module in super().find_package_modules(package, package_dir)
            if module[:2] != ("aegaeon", "energy")
        ]

    def _profile_files(self):
        return sorted((ROOT_DIR / "profiles").rglob("*.json"))

    def run(self):
        super().run()
        for source in self._profile_files():
            target = Path(self.build_lib) / "aegaeon" / "_profiles" / source.relative_to(ROOT_DIR / "profiles")
            self.mkpath(str(target.parent))
            self.copy_file(str(source), str(target))
        target = Path(self.build_lib) / "aegaeon" / "_build_info.json"
        self.mkpath(str(target.parent))
        target.write_text(json.dumps({
            "cuda": ROUTE,
            "minimum_glibc": "2.31" if NATIVE else None,
            "torch": "2.11.0" if ROUTE != "core" else None,
            "vllm": "0.26.0" if ROUTE != "core" else None,
            "native": NATIVE,
            "quick_model_loader": NATIVE,
            "pypi_release": RELEASE and not SDIST,
        }, indent=2) + "\n", encoding="utf-8")

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        outputs.extend(str(Path(self.build_lib) / "aegaeon" / "_profiles" /
                           source.relative_to(ROOT_DIR / "profiles"))
                       for source in self._profile_files())
        outputs.append(str(Path(self.build_lib) / "aegaeon" / "_build_info.json"))
        return outputs


class BuildCUDA(build_ext):
    """Load Torch and create CUDAExtension only when native compilation runs."""

    def run(self):
        expected = {"cu129": "12.9", "cu130": "13.0"}[ROUTE]
        prefix = Path(sys.executable).resolve().parent.parent
        toolkit = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if toolkit is None and (prefix / "bin/nvcc").is_file():
            toolkit = str(prefix)
        if toolkit is None:
            nvcc = shutil.which("nvcc")
            if nvcc:
                toolkit = str(Path(nvcc).resolve().parent.parent)
        if toolkit is None or not (Path(toolkit) / "bin/nvcc").is_file():
            raise RuntimeError(f"Selected {ROUTE} needs CUDA Toolkit {expected}. "
                               "Install the matching Toolkit and set CUDA_HOME before building; "
                               "see docs/en/installation.md.")
        version = subprocess.run([str(Path(toolkit) / "bin/nvcc"), "--version"],
                                 capture_output=True, text=True, check=True, timeout=10).stdout
        if not re.search(r"release " + re.escape(expected) + r"(?:,|\s)", version):
            raise RuntimeError(f"Selected {ROUTE} requires CUDA {expected}; CUDA_HOME={toolkit} has {version.strip()}")
        os.environ["CUDA_HOME"] = toolkit
        import torch
        if torch.__version__.split("+")[0] != "2.11.0" or torch.version.cuda != expected:
            raise RuntimeError(f"Selected {ROUTE} needs PyTorch 2.11.0 / CUDA {expected}; "
                               f"found {torch.__version__} / {torch.version.cuda}. "
                               "Prepare the matching backend before --no-build-isolation.")
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension
        cccl = Path(toolkit) / "include" / "cccl"
        extensions = [CUDAExtension(
            name="aegaeon.ops", sources=["ops/ops.cpp", "ops/swap.cu"],
            include_dirs=[str(cccl)] if cccl.is_dir() else [],
        )]
        # NVIDIA's pip Toolkit contains the versioned runtime but may omit
        # the unversioned linker symlink. Link its SONAME-bearing object directly.
        extension = extensions[0]
        if not any((Path(directory) / "libcudart.so").is_file()
                   for directory in extension.library_dirs):
            versioned = [Path(toolkit) / directory / ("libcudart.so." + expected.split(".")[0])
                         for directory in ("lib64", "lib")]
            runtime = next((path for path in versioned if path.is_file()), None)
            if runtime is None:
                raise RuntimeError("CUDA Toolkit is missing its libcudart shared library.")
            extension.libraries.remove("cudart")
            extension.extra_objects = list(extension.extra_objects or []) + [str(runtime)]
        self.distribution.ext_modules = extensions
        self.extensions = extensions
        command = BuildExtension(self.distribution)
        command.ensure_finalized()
        for name in ("build_lib", "build_temp", "inplace", "force", "debug", "compiler",
                     "parallel", "dry_run"):
            setattr(command, name, getattr(self, name))
        command.run()


extensions = ([Extension("aegaeon.ops", sources=["ops/ops.cpp", "ops/swap.cu"])]
              if NATIVE else [])
rust_extensions = []
if NATIVE:
    from setuptools_rust import Binding, RustExtension
    rust_extensions = [RustExtension(
        "quick_model_loader._rlib",
        path="quick_model_loader/Cargo.toml",
        binding=Binding.PyO3,
        py_limited_api=True,
        debug=False,
    )]

loader_packages = find_packages(where="quick_model_loader",
                               include=["quick_model_loader", "quick_model_loader.*"])
setup(
    name="aegaeon",
    version="0.1.0",
    author="PKUSYS",
    description="Aegaeon: A multi-LLM inference engine that multiplexes requests at the token level.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    license_files=["LICENSE"],
    url="https://github.com/pkusys/aegaeon",
    packages=find_packages(include=["aegaeon", "aegaeon.*"]) + loader_packages,
    package_dir={"quick_model_loader": "quick_model_loader/quick_model_loader"},
    python_requires=">=3.10,<3.11",
    install_requires=get_requirements(),
    entry_points={"console_scripts": ["aegaeon=aegaeon.cli:main"]},
    ext_modules=extensions,
    rust_extensions=rust_extensions,
    cmdclass={"build_py": BuildPy, "build_ext": BuildCUDA},
    # Keep CUDA 12.9 wheel filenames distinguishable from CUDA 13 artifacts.
    options={"bdist_wheel": {"build_number": "1cu129"}} if NATIVE and ROUTE == "cu129" else {},
    zip_safe=False,
)
