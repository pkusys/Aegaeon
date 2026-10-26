"""CUDA installation policy, usable before importing any serving dependencies."""

import argparse
from dataclasses import dataclass
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
from urllib.parse import unquote, urlsplit


class CUDASelectionError(ValueError):
    pass


@dataclass(frozen=True)
class BackendPlan:
    route: str
    reason: str


def release_mode():
    """Release wheels use an explicit backend with registry version requirements."""
    value = os.environ.get("AEGAEON_RELEASE", "0")
    if value not in {"0", "1"}:
        raise CUDASelectionError("AEGAEON_RELEASE must be 0 or 1")
    return value == "1"


def python_abi_tag(version=None):
    """Validate Aegaeon's Python support policy and return its CPython ABI tag."""
    version = tuple(sys.version_info[:2] if version is None else version[:2])
    if version != (3, 10):
        raise CUDASelectionError(
            "Aegaeon requires standard CPython 3.10 (>=3.10,<3.11). "
            "Create and activate a Python 3.10 environment first."
        )
    if platform.python_implementation() != "CPython":
        raise CUDASelectionError("The GPU wheels require standard CPython with the GIL enabled.")
    if sysconfig.get_config_var("Py_GIL_DISABLED") in (1, "1", True):
        raise CUDASelectionError("The GPU wheels do not support free-threaded CPython; use a GIL-enabled interpreter.")
    return "cp" + str(version[0]) + str(version[1])


def cuda129_wheels(python_version=None):
    return dict(item.split(" @ ", 1) for item in
                runtime_requirements("cu129", python_version=python_version)
                if " @ " in item)


def select_backend(requested=None):
    """Use the user's build target without inspecting the build host's GPUs."""
    if requested in {None, "", "auto"}:
        raise CUDASelectionError(
            "Choose a CUDA target explicitly: set AEGAEON_CUDA=cu129 or "
            "AEGAEON_CUDA=cu130 and install the matching requirements-cu129.txt "
            "or requirements-cu130.txt before building."
        )
    if requested == "core":
        return BackendPlan("core", "Python/resource package without CUDA runtime dependencies")
    if requested not in {"cu129", "cu130"}:
        raise CUDASelectionError(f"Unsupported CUDA selection: {requested}")
    return BackendPlan(requested, "Explicit CUDA source-build target")


def runtime_requirements(route, *, python_version=None, registry=False):
    """Read the same CUDA pins used by the user's explicit pip command."""
    python_abi_tag(python_version)
    if route == "core":
        return []
    if route not in {"cu129", "cu130"}:
        raise CUDASelectionError(f"Unsupported CUDA selection: {route}")
    manifest = Path(__file__).resolve().parent / ("requirements-" + route + ".txt")
    requirements = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith(("#", "-r "))]
    if registry:
        # PyPI metadata cannot add wheel indexes or direct URL dependencies.
        # Preserve each selected wheel's CUDA ABI in its version constraint.
        result = []
        for requirement in requirements:
            name, separator, url = requirement.partition(" @ ")
            if separator:
                filename = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
                result.append(name + "==" + filename.split("-")[1])
            else:
                result.append(requirement)
        return result
    return requirements


def validate_installed_cupy(route, *, installed=None):
    """Reject conflicting CuPy distributions before the selected backend is installed."""
    if route == "core":
        return
    expected = {"cu129": "cupy-cuda12x", "cu130": "cupy-cuda13x"}.get(route)
    if expected is None:
        raise CUDASelectionError(f"Unsupported CUDA selection: {route}")
    if installed is None:
        # pip isolation uses PYTHONPATH to hide the invoking interpreter's packages.
        # -E ignores that build environment while retaining normal user-site policy.
        # This is a read-only probe; never install packages from a build hook.
        script = """import importlib.metadata as metadata, json
found = {}
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name", "").lower().replace("_", "-").replace(".", "-")
    if name == "cupy" or name.startswith("cupy-cuda"):
        found[name] = distribution.version
print(json.dumps(found))
"""
        try:
            result = subprocess.run([sys.executable, "-E", "-c", script],
                                    cwd=str(Path(sys.executable).parent),
                                    capture_output=True, text=True, check=True, timeout=10)
            installed = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise CUDASelectionError(f"Cannot inspect the current Python environment's CuPy packages: {exc}") from exc
    conflicts = {name: version for name, version in installed.items() if name != expected}
    if conflicts:
        details = ", ".join(name + "==" + str(version) for name, version in sorted(conflicts.items()))
        raise CUDASelectionError(f"Selected {route} requires only {expected}; the current Python "
                                 f"environment contains conflicting CuPy packages: {details}. "
                                 "Use a fresh environment when changing CUDA routes.")


def build_requirements(route, *, python_version=None):
    return [item for item in runtime_requirements(route, python_version=python_version) if item.startswith("torch @ ")
            or item.startswith("torch===")]


def validate_install_platform(route=None):
    machine = platform.machine()
    if platform.system() != "Linux" or machine not in {"x86_64", "aarch64"}:
        raise CUDASelectionError("CUDA source installation supports Linux x86_64 and aarch64.")
    if machine == "aarch64" and route != "cu130":
        raise CUDASelectionError(
            "Linux aarch64 source installation requires AEGAEON_CUDA=cu130; "
            "requirements-cu129.txt contains x86_64-only wheel URLs."
        )
    python_abi_tag()
    libc = os.confstr("CS_GNU_LIBC_VERSION") or ""
    match = re.search(r"([0-9]+)\.([0-9]+)$", libc)
    if not match or tuple(map(int, match.groups())) < (2, 28):
        raise CUDASelectionError(f"The CUDA wheels require glibc >=2.28; found {libc or 'unknown'}.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", choices=("cu129", "cu130", "core"), required=True)
    parser.add_argument("--format", choices=("json", "tsv", "requirements"), default="json")
    args = parser.parse_args(argv)
    try:
        python_abi_tag()
        plan = select_backend(args.cuda)
        if plan.route != "core":
            validate_install_platform(plan.route)
    except CUDASelectionError as exc:
        parser.exit(1, str(exc) + "\n")
    if args.format == "requirements":
        print("\n".join(runtime_requirements(plan.route)))
    elif args.format == "tsv":
        print(plan.route + "\t" + plan.reason)
    else:
        print(json.dumps({"route": plan.route, "reason": plan.reason,
                          "requirements": runtime_requirements(plan.route)}, indent=2))


if __name__ == "__main__":
    main()
