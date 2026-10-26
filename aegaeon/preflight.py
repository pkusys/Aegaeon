"""Runtime diagnostics that can be invoked without importing the serving stack."""

from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path
import sys


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    severity: str = "error"

    @property
    def blocking(self) -> bool:
        return not self.ok and self.severity != "warning"

    @property
    def status(self) -> str:
        return "OK" if self.ok else ("FAIL" if self.blocking else "WARN")


def check_runtime(*, check_cuda: bool = False, config_path=None,
                  models=None, tensor_parallel_size=None) -> list[CheckResult]:
    results = []
    modules = {}
    dependencies = (
        ("torch", "Install torch==2.11.0 with the matching CUDA wheel.", "2.11.0"),
        ("vllm", "Install vllm==0.26.0 with its matching native dependencies.", "0.26.0"),
        ("ray", "Install the dependencies in requirements.txt.", None),
        ("yaml", "Install PyYAML from requirements.txt.", None),
        ("cupy", "Install cupy-cuda12x for CUDA 12 or cupy-cuda13x for CUDA 13, matching the installed PyTorch/vLLM stack.", None),
        ("fastapi", "Install the dependencies in requirements.txt.", None),
        ("uvicorn", "Install the dependencies in requirements.txt.", None),
        ("aegaeon.ops", "Reinstall the Aegaeon wheel matching this Python/PyTorch/CUDA environment. "
         "For a source checkout, rebuild with python -m pip install --no-build-isolation -e .", None),
        (
            "quick_model_loader._rlib",
            "Reinstall the Aegaeon wheel matching this Python/PyTorch/CUDA environment; "
            "it includes quick_model_loader. For a source checkout, install Rust/Cargo and "
            "rebuild with python -m pip install --no-build-isolation -e . from the repository root.",
            None,
        ),
    )
    for name, hint, expected in dependencies:
        try:
            module = import_module(name)
            modules[name] = module
            version = getattr(module, "__version__", None)
            if expected and str(version).split("+", 1)[0] != expected:
                raise RuntimeError(f"expected {expected}, found {version}")
            if name == "aegaeon.ops" and not callable(getattr(module, "swap", None)):
                raise RuntimeError("native extension does not export callable swap")
            results.append(CheckResult(name, True, str(version or "import OK")))
        except Exception as exc:
            results.append(CheckResult(name, False, f"{type(exc).__name__}: {exc}. {hint}"))

    build_info_path = Path(__file__).resolve().parent / "_build_info.json"
    if build_info_path.is_file():
        try:
            build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
            route = build_info["cuda"]
            expected_cuda = {"cu129": "12.9", "cu130": "13.0"}.get(route)
            if route != "core" and expected_cuda is None:
                raise ValueError(f"unknown wheel CUDA route {route!r}")
            if expected_cuda is not None:
                torch = modules.get("torch")
                actual_cuda = getattr(getattr(torch, "version", None), "cuda", None)
                if actual_cuda != expected_cuda:
                    raise RuntimeError(
                        f"Aegaeon wheel was built for {route} / CUDA {expected_cuda}, "
                        f"but torch.version.cuda is {actual_cuda!r}"
                    )
                results.append(CheckResult(
                    "Aegaeon wheel CUDA ABI", True,
                    f"{route}: torch.version.cuda={actual_cuda}",
                ))
        except Exception as exc:
            results.append(CheckResult(
                "Aegaeon wheel CUDA ABI", False,
                f"{type(exc).__name__}: {exc}. Reinstall the Aegaeon wheel "
                "matching the installed PyTorch/CUDA environment.",
            ))

    if check_cuda:
        torch = modules.get("torch")
        try:
            if torch is None:
                raise RuntimeError("torch import failed; fix the dependency check first")
            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA device is available in this process")
            # Availability alone does not exercise the installed CUDA runtime.
            value = (torch.ones(1, device="cuda") + 1).item()
            torch.cuda.synchronize()
            if value != 2:
                raise RuntimeError(f"unexpected CUDA tensor result: {value}")
            results.append(CheckResult("CUDA tensor operation", True, f"CUDA {torch.version.cuda}"))
        except Exception as exc:
            results.append(CheckResult(
                "CUDA tensor operation", False,
                f"{type(exc).__name__}: {exc}. Check nvidia-smi, torch.version.cuda, "
                "and the driver requirements of the installed wheels.",
            ))
        try:
            cupy = modules.get("cupy")
            if cupy is None:
                raise RuntimeError("cupy import failed; fix the dependency check first")
            if cupy.ones(1).sum().item() != 1:
                raise RuntimeError("unexpected CuPy tensor result")
            cupy.cuda.get_current_stream().synchronize()
            results.append(CheckResult("CuPy CUDA operation", True, "tensor operation OK"))
        except Exception as exc:
            results.append(CheckResult(
                "CuPy CUDA operation", False,
                f"{type(exc).__name__}: {exc}. Match the CuPy CUDA package and Toolkit "
                "to the installed PyTorch/vLLM stack.",
            ))
        try:
            import_module("vllm._C_stable_libtorch")
            results.append(CheckResult("vllm._C_stable_libtorch", True, "native import OK"))
        except Exception as exc:
            results.append(CheckResult(
                "vllm._C_stable_libtorch", False,
                f"{type(exc).__name__}: {exc}. Reinstall the complete matching vLLM/PyTorch "
                "wheel combination; replacing only torch does not rebuild vLLM.",
            ))
    cuda_devices = None
    torch = modules.get("torch")
    if check_cuda and config_path is not None and torch is not None:
        try:
            cuda_devices = [torch.cuda.get_device_name(index)
                            for index in range(torch.cuda.device_count())]
        except Exception as exc:
            results.append(CheckResult("CUDA device inventory", False,
                                       f"{type(exc).__name__}: {exc}"))
    from .preflight_config import check_configuration
    from .preflight_host import check_host

    configuration, budget = check_configuration(
        config_path, models=models, tensor_parallel_size=tensor_parallel_size,
        cuda_devices=cuda_devices,
    )
    results.extend(configuration)
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    results.extend(check_host(cuda_version=cuda_version,
                              memory_budget_bytes=budget, check_cuda=check_cuda))
    return results


def ensure_runtime(*, config_path=None, models=None, tensor_parallel_size=None) -> None:
    results = check_runtime(config_path=config_path, models=models,
                            tensor_parallel_size=tensor_parallel_size)
    for result in results:
        if not result.ok and not result.blocking:
            print(f"WARN {result.name}: {result.detail}", file=sys.stderr)
    failures = [result for result in results if result.blocking]
    if failures:
        raise RuntimeError("Runtime preflight failed:\n" + "\n".join(
            f"  {result.name}: {result.detail}" for result in failures
        ))
