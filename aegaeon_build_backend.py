"""PEP 517 hooks for an explicitly selected CUDA source-build target."""

from contextlib import contextmanager
import json
import os
import email
import platform
import zipfile
import subprocess
import tempfile
from pathlib import Path

from aegaeon_cuda import (BackendPlan, CUDASelectionError, build_requirements,
                         python_abi_tag, release_mode, runtime_requirements,
                         select_backend, validate_install_platform,
                         validate_installed_cupy)

# Reject unsupported interpreters before invoking setuptools or resolving dependencies.
python_abi_tag()
from setuptools import build_meta

RECEIPT = "aegaeon-cuda-selection.json"


def _native_enabled():
    value = os.environ.get("AEGAEON_BUILD_CUDA", "1")
    if value not in {"0", "1"}:
        raise ValueError("AEGAEON_BUILD_CUDA must be 0 or 1")
    return value == "1"


def _plan():
    python_abi_tag()
    requested = os.environ.get("AEGAEON_CUDA")
    if release_mode():
        if requested not in {"cu129", "cu130"} or not _native_enabled():
            raise CUDASelectionError("PyPI release wheels require explicit AEGAEON_CUDA=cu129 or cu130 "
                                     "and AEGAEON_BUILD_CUDA=1; auto/core are not release targets.")
        # A build host need not have a GPU or a driver capable of executing the
        # target. Compilation still verifies the actual Torch and CUDA Toolkit.
        plan = BackendPlan(requested, "Explicit PyPI wheel target")
    else:
        plan = select_backend(requested)
    if plan.route == "core":
        if _native_enabled():
            raise CUDASelectionError("AEGAEON_CUDA=core requires AEGAEON_BUILD_CUDA=0.")
    else:
        validate_install_platform(plan.route)
        validate_installed_cupy(plan.route)
    return plan


@contextmanager
def _selected(plan, *, sdist=False):
    values = {"AEGAEON_RESOLVED_CUDA": plan.route,
              "AEGAEON_BUILDING_SDIST": "1" if sdist else "0"}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def get_requires_for_build_wheel(config_settings=None):
    plan = _plan()
    # Do not run setup.py or import Torch in the initial, isolated bootstrap.
    return build_requirements(plan.route) if _native_enabled() else []


def get_requires_for_build_editable(config_settings=None):
    return get_requires_for_build_wheel(config_settings)


def _metadata(hook, metadata_directory, config_settings):
    plan = _plan()
    with _selected(plan):
        relative = hook(metadata_directory, config_settings)
    receipt = {"route": plan.route, "reason": plan.reason, "native": _native_enabled(),
               "release": release_mode()}
    (Path(metadata_directory) / relative / RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
    return relative


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _metadata(build_meta.prepare_metadata_for_build_wheel, metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return _metadata(build_meta.prepare_metadata_for_build_editable, metadata_directory, config_settings)


def _build_plan(metadata_directory):
    python_abi_tag()
    if metadata_directory is None:
        return _plan()
    directory = Path(metadata_directory)
    receipts = ([directory / RECEIPT] if (directory / RECEIPT).is_file()
                else list(directory.glob("*.dist-info/" + RECEIPT)))
    if len(receipts) != 1:
        raise CUDASelectionError("Prepared CUDA metadata is missing its selection receipt; prepare metadata again.")
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    if receipt.get("release", False) != release_mode():
        raise CUDASelectionError("AEGAEON_RELEASE changed after metadata preparation.")
    if receipt["native"] != _native_enabled():
        raise CUDASelectionError("AEGAEON_BUILD_CUDA changed after metadata preparation.")
    requested = os.environ.get("AEGAEON_CUDA")
    if requested != receipt["route"]:
        raise CUDASelectionError("AEGAEON_CUDA changed after metadata preparation.")
    if release_mode() and (requested not in {"cu129", "cu130"}
                           or receipt["route"] != requested or not _native_enabled()):
        raise CUDASelectionError("Prepared metadata is not a complete native release for the requested CUDA target.")
    plan = BackendPlan(receipt["route"], receipt["reason"])
    if plan.route != "core":
        validate_install_platform(plan.route)
    validate_installed_cupy(plan.route)
    return plan


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    plan = _build_plan(metadata_directory)
    if not _native_enabled():
        with _selected(plan):
            return build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)
    destination = Path(wheel_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Never leave an unverified raw Linux wheel in the output directory.
    with tempfile.TemporaryDirectory(prefix="aegaeon-wheel-", dir=destination.parent) as temporary:
        work = Path(temporary)
        raw = work / "raw"
        repaired = work / "repaired"
        raw.mkdir()
        repaired.mkdir()
        with _selected(plan):
            filename = build_meta.build_wheel(str(raw), config_settings, metadata_directory)
        candidate = _repair_native_wheel(raw / filename, repaired, plan.route)
        if release_mode():
            _check_release_wheel(candidate, plan.route)
        target = destination / candidate.name
        os.replace(candidate, target)
        return target.name


def _manylinux_platform():
    machine = platform.machine()
    if platform.system() != "Linux" or machine not in {"x86_64", "aarch64"}:
        raise CUDASelectionError("Native wheels require a Linux x86_64 or aarch64 build host.")
    return "manylinux_2_31_" + machine


def _validate_wheel_architecture(path):
    """Check the source tags before repair can rename a wheel for this host."""
    from packaging.tags import parse_tag
    from packaging.utils import parse_wheel_filename
    machine = platform.machine()
    _manylinux_platform()
    _, _, _, filename_tags = parse_wheel_filename(Path(path).name)
    with zipfile.ZipFile(path) as archive:
        wheel = email.message_from_bytes(
            archive.read(next(name for name in archive.namelist()
                              if name.endswith(".dist-info/WHEEL"))))
    metadata_tags = set().union(*(parse_tag(value) for value in wheel.get_all("Tag", [])))
    for tags in (filename_tags, metadata_tags):
        if not tags or any(not tag.platform.endswith("_" + machine) for tag in tags):
            raise CUDASelectionError(
                f"Native wheel architecture does not match the {machine} build host: {Path(path).name}"
            )


def _repair_native_wheel(raw_path, wheel_directory, route):
    """Use dependency-provided Torch/CUDA libraries without build-host paths."""
    from wheel.wheelfile import WheelFile
    validate_install_platform(route)
    _validate_wheel_architecture(raw_path)
    cuda_directory = "cuda_runtime" if route == "cu129" else "cu13"
    rpath = "$ORIGIN/../torch/lib:$ORIGIN/../nvidia/" + cuda_directory + "/lib"
    with tempfile.TemporaryDirectory(prefix="aegaeon-elf-", dir=raw_path.parent) as temporary:
        work = Path(temporary)
        unpacked = work / "unpacked"
        unpacked.mkdir()
        with zipfile.ZipFile(raw_path) as archive:
            for name in archive.namelist():
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise CUDASelectionError("Unexpected wheel archive path: " + name)
            archive.extractall(unpacked)
        for native in unpacked.rglob("*.so"):
            if native.parent.name == "aegaeon" and native.name.startswith("ops."):
                subprocess.run(["patchelf", "--set-rpath", rpath, str(native)], check=True)
            else:
                subprocess.run(["patchelf", "--remove-rpath", str(native)], check=True)
        patched = work / raw_path.name
        # WheelFile regenerates RECORD hashes after the ELF changes.
        with WheelFile(str(patched), "w") as archive:
            archive.write_files(str(unpacked))
        command = ["auditwheel", "repair", "--plat", _manylinux_platform(),
                   "--wheel-dir", str(wheel_directory)]
        for library in ("libc10.so", "libc10_cuda.so", "libtorch.so",
                        "libtorch_cpu.so", "libtorch_cuda.so", "libtorch_python.so",
                        "libcudart.so.12", "libcudart.so.13"):
            command.extend(["--exclude", library])
        command.append(str(patched))
        subprocess.run(command, check=True)
    candidates = list(Path(wheel_directory).glob("*.whl"))
    if len(candidates) != 1:
        raise CUDASelectionError("Expected one repaired native wheel, found " + str(len(candidates)))
    return candidates[0]

def _check_release_wheel(path, expected_route):
    """Reject incomplete binaries and dependencies unsuitable for PyPI."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    if expected_route not in {"cu129", "cu130"}:
        raise CUDASelectionError("Unknown release CUDA target: " + str(expected_route))
    validate_install_platform(expected_route)
    _validate_wheel_architecture(path)
    expected_platform = _manylinux_platform()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = email.message_from_bytes(archive.read(metadata_name))
        wheel_metadata = email.message_from_bytes(
            archive.read(next(name for name in names if name.endswith(".dist-info/WHEEL"))))
        if not any(tag.split("-")[-1] == expected_platform
                   for tag in wheel_metadata.get_all("Tag", [])):
            raise CUDASelectionError(f"Release wheel must have a verified {expected_platform} tag.")
        requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
        if any(requirement.url for requirement in requirements):
            raise CUDASelectionError("PyPI release metadata must not contain direct URL dependencies.")
        for value in runtime_requirements(expected_route, registry=True):
            expected = Requirement(value)
            matching = [requirement for requirement in requirements
                        if canonicalize_name(requirement.name) == canonicalize_name(expected.name)]
            if (len(matching) != 1 or matching[0].marker is not None
                    or matching[0].extras
                    or str(matching[0].specifier) != str(expected.specifier)):
                raise CUDASelectionError(
                    f"Release wheel backend dependency mismatch: expected unconditional {expected}."
                )
        conflicting_cupy = {"cupy", "cupy-cuda13x" if expected_route == "cu129" else "cupy-cuda12x"}
        if any(canonicalize_name(requirement.name) in conflicting_cupy for requirement in requirements):
            raise CUDASelectionError("Release wheel backend dependency mismatch: conflicting CuPy CUDA package.")
        if any(canonicalize_name(requirement.name).startswith("aegaeon-runtime")
               for requirement in requirements):
            raise CUDASelectionError("Release wheels use separate upstream dependencies, not a runtime companion.")
        if any("_vendor" in Path(name).parts or name.endswith(".pth")
               or name.split("/", 1)[0] in {"torch", "vllm", "nvidia"}
               for name in names):
            raise CUDASelectionError("Release wheel contains a vendored runtime payload.")
        for prefix in ("aegaeon/ops.", "quick_model_loader/_rlib."):
            if not any(name.startswith(prefix) and name.endswith(".so") for name in names):
                raise CUDASelectionError("Release wheel is missing compiled extension: " + prefix)
        if not any(name.startswith("aegaeon/_profiles/") and name.endswith(".json") for name in names):
            raise CUDASelectionError("Release wheel is missing bundled latency profiles.")
        info = json.loads(archive.read("aegaeon/_build_info.json"))
        if info.get("cuda") != expected_route or not all(
                info.get(key) for key in ("native", "quick_model_loader", "pypi_release")):
            raise CUDASelectionError(
                f"Release wheel build information does not describe a complete {expected_route} build."
            )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    plan = _build_plan(metadata_directory)
    with _selected(plan):
        return build_meta.build_editable(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_sdist(config_settings=None):
    python_abi_tag()
    return []


def build_sdist(sdist_directory, config_settings=None):
    python_abi_tag()
    # Source archives retain the base version; the target source build selects CUDA.
    with _selected(BackendPlan("core", "Source archive"), sdist=True):
        return build_meta.build_sdist(sdist_directory, config_settings)
