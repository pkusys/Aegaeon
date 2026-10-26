import platform
import sys
import sysconfig

# Keep the stable ABI build while restricting installation to the serving runtime.
if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 10):
    raise RuntimeError("quick_model_loader requires standard CPython 3.10 (>=3.10,<3.11).")
if sysconfig.get_config_var("Py_GIL_DISABLED"):
    raise RuntimeError(
        "quick_model_loader requires standard CPython with the GIL; "
        "free-threaded Python does not support its stable ABI."
    )

from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    name="quick_model_loader",
    version="0.0.2",
    packages=find_packages(include=["quick_model_loader*"]),
    install_requires=[
        "torch>=2.0.0",
        "safetensors>=0.4.3",
        "numpy",
        "pytest",
    ],
    # Select one CuPy distribution matching the runtime; both share the cupy namespace.
    # CUDA 12 is tested; the optional CUDA 13 quickstart is not end-to-end tested.
    extras_require={
        "cu12": ["cupy-cuda12x"],
        "cu13": ["cupy-cuda13x"],
    },
    rust_extensions=[
        RustExtension(
            "quick_model_loader._rlib",
            # ^-- The last part of the name (e.g. "_lib") has to match lib.name
            #     in Cargo.toml and the function name in the `.rs` file,
            #     but you can add a prefix to nest it inside of a Python package.
            path="Cargo.toml",  # Default value, can be omitted
            binding=Binding.PyO3,  # Default value, can be omitted
            py_limited_api=True,
        )
    ],
    python_requires=">=3.10,<3.11",
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
