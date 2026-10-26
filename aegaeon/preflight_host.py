"""Local host checks without importing Torch, Ray, or the serving stack."""

import ctypes
import mmap
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile

from .preflight import CheckResult


def _warning(name: str, detail: str) -> CheckResult:
    return CheckResult(name, False, detail, severity="warning")


def _command_version(command: list[str]) -> str:
    result = subprocess.run(
        [*command, "--version"], capture_output=True, text=True,
        check=True, timeout=10,
    )
    output = (result.stdout or result.stderr).strip()
    if not output:
        raise RuntimeError("--version returned no version information")
    return output


def _check_toolchain(cuda_version: str | None) -> list[CheckResult]:
    results = []
    # Keep source-build selection aligned with setup.py's BuildCUDA.
    toolkit = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    prefix = Path(sys.executable).resolve().parent.parent
    if toolkit is None and (prefix / "bin/nvcc").is_file():
        toolkit = str(prefix)
    if toolkit is None:
        nvcc_on_path = shutil.which("nvcc")
        if nvcc_on_path:
            toolkit = str(Path(nvcc_on_path).resolve().parent.parent)
    try:
        if toolkit is None:
            raise RuntimeError("nvcc was not found in CUDA_HOME/CUDA_PATH, the Python environment, or PATH")
        nvcc = Path(toolkit) / "bin/nvcc"
        version = _command_version([str(nvcc)])
        match = re.search(r"release\s+(\d+\.\d+)(?:,|\s|$)", version)
        if not match:
            raise RuntimeError(f"could not parse nvcc release at {nvcc}: {version}")
        found = match.group(1)
        expected = re.match(r"\d+\.\d+", str(cuda_version or ""))
        if expected is None:
            results.append(CheckResult(
                "CUDA toolchain", False, f"nvcc {found} at {nvcc}; PyTorch CUDA version unavailable, so compatibility is unverified. Fix the runtime dependency check first."
            ))
        elif found != expected.group():
            results.append(CheckResult(
                "CUDA toolchain", False, f"nvcc {found} at {nvcc} does not match PyTorch CUDA {expected.group()}. "
                "Set CUDA_HOME to a matching Toolkit for native builds and serving-time JIT compilation."
            ))
        else:
            results.append(CheckResult("CUDA toolchain", True, f"nvcc {found} at {nvcc}; matches PyTorch CUDA {expected.group()}"))
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        results.append(CheckResult(
            "CUDA toolchain", False, f"{type(exc).__name__}: {exc}. Install the matching CUDA Toolkit or set CUDA_HOME "
            "for native builds and serving-time JIT compilation."
        ))

    try:
        configured = os.environ.get("CXX")
        command = shlex.split(configured) if configured else [shutil.which("c++") or shutil.which("g++") or shutil.which("clang++") or ""]
        if not command or not command[0]:
            raise RuntimeError("no C++ compiler found in CXX or PATH")
        version = _command_version(command)
        results.append(CheckResult("C++ toolchain", True, f"{shlex.join(command)}: {version.splitlines()[0]}"))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        results.append(CheckResult("C++ toolchain", False, f"{type(exc).__name__}: {exc}. Install a C++ compiler or set CXX for native builds and serving-time JIT compilation."))

    for command in ("rustc", "cargo"):
        try:
            executable = shutil.which(command)
            if not executable:
                raise RuntimeError(f"{command} is absent from PATH")
            version = _command_version([executable])
            results.append(CheckResult(command, True, version.splitlines()[0]))
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            results.append(_warning(command, f"{type(exc).__name__}: {exc}. Required to build quick_model_loader, but not to load an installed native extension."))
    return results


def _probe_shared_mapping(directory: Path) -> None:
    # Never create or resize the actual aegaeon cache files.
    with tempfile.TemporaryFile(prefix="aegaeon-doctor-", dir=directory) as handle:
        handle.truncate(mmap.PAGESIZE)
        with mmap.mmap(handle.fileno(), mmap.PAGESIZE, access=mmap.ACCESS_WRITE) as mapping:
            mapping[:8] = b"aegaeon!"
            mapping.flush()
            if mapping[:8] != b"aegaeon!":
                raise RuntimeError("shared mmap did not retain the probe bytes")


def _check_shared_memory(directory: Path, budget: int | None,
                         reusable_bytes: int | None = 0) -> list[CheckResult]:
    try:
        if not directory.is_dir():
            raise FileNotFoundError(f"shared-memory directory does not exist: {directory}")
        usage = os.statvfs(directory)
        available = usage.f_bavail * usage.f_frsize
        if available < mmap.PAGESIZE:
            possible_reuse = reusable_bytes is None or (
                reusable_bytes > 0 and (budget is None or max(0, budget - reusable_bytes) <= available)
            )
            if not possible_reuse:
                raise OSError(f"only {available} bytes available; cannot safely allocate a {mmap.PAGESIZE}-byte mmap probe")
            result = _warning("shared-memory mmap", f"{directory}: only {available} bytes available; the {mmap.PAGESIZE}-byte mmap probe was not attempted. Reuse of existing cache allocation might suffice, but mmap access is unverified.")
        else:
            _probe_shared_mapping(directory)
            result = CheckResult("shared-memory mmap", True, f"{directory}: local {mmap.PAGESIZE}-byte shared mmap read/write succeeded and was removed")
    except (OSError, ValueError, RuntimeError) as exc:
        return [CheckResult("shared-memory mmap", False, f"{type(exc).__name__}: {exc}. Configure a writable AEGAEON_SHM_DIR or node shm_dir on this host.")]

    if budget is None:
        capacity = _warning("shared-memory capacity", f"{directory}: {available} bytes available locally; no configuration budget provided. Use doctor --config to compare CPU slabs plus model cache.")
    elif available < budget:
        detail = f"{directory}: {available} bytes available locally for {budget} configured CPU slab plus model cache bytes. "
        if reusable_bytes is None:
            capacity = _warning("shared-memory capacity", detail + "Existing cache allocation could not be inspected reliably. Fresh allocation exceeds free space, but reuse capacity is unverified.")
        elif max(0, budget - reusable_bytes) <= available:
            capacity = _warning("shared-memory capacity", detail + f"Existing cache files have {reusable_bytes} allocated bytes that might be reused; the minimum additional allocation could be {max(0, budget - reusable_bytes)} bytes. Reuse depends on each cache's required size and ownership/use; it has not been confirmed.")
        else:
            capacity = CheckResult("shared-memory capacity", False, detail + f"Even if all {reusable_bytes} existing allocated cache bytes were reused, at least {max(0, budget - reusable_bytes)} additional bytes would be needed. Reduce the cache budget, free space, or enlarge the shared-memory mount.")
    else:
        capacity = CheckResult("shared-memory capacity", True, f"{directory}: {available} bytes available locally for {budget} configured bytes; this is a snapshot, not a reservation")
    return [result, capacity]


def _check_memlock(budget: int | None) -> CheckResult:
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        describe = lambda value: "unlimited" if value == resource.RLIM_INFINITY else f"{value} bytes"
        detail = f"Local RLIMIT_MEMLOCK soft={describe(soft)}, hard={describe(hard)}. "
        if budget is None:
            return _warning("memlock", detail + "No configuration budget provided; capacity is unverified. CUDA host registration is checked only with --check-cuda.")
        detail += f"Configured pinned cache budget={budget} bytes. "
        if soft != resource.RLIM_INFINITY and soft < budget:
            return _warning("memlock", detail + "The soft limit is below the budget. Review ulimit -l or container memlock settings; this limit alone does not determine CUDA registration capacity. Use --check-cuda for a small registration probe.")
        return CheckResult("memlock", True, detail + "Limits permit this budget; a limit check cannot guarantee full-budget CUDA registration.")
    except (ImportError, AttributeError, OSError, ValueError) as exc:
        return _warning("memlock", f"{type(exc).__name__}: {exc}. Cannot inspect the local locked-memory limit on this platform.")


def _check_cache_files(directory: Path) -> list[CheckResult]:
    results = []
    for name in ("aegaeon_cpu_cache", "aegaeon_model_cache"):
        path = directory / name
        try:
            path.lstat()
        except FileNotFoundError:
            results.append(CheckResult(f"shared-memory file {name}", True, f"{path}: no existing local cache file"))
            continue
        except OSError as exc:
            results.append(CheckResult(f"shared-memory file {name}", False, f"Cannot inspect {path}: {exc}"))
            continue
        if not path.is_file():
            results.append(CheckResult(f"shared-memory file {name}", False, f"{path} exists but does not resolve to a regular file. Choose another node shm_dir; doctor has not changed it."))
        elif not os.access(path, os.R_OK | os.W_OK):
            results.append(CheckResult(f"shared-memory file {name}", False, f"{path} exists but this process cannot read and write it. Fix ownership/permissions or choose another node shm_dir; doctor has not changed it."))
        else:
            results.append(_warning(f"shared-memory file {name}", f"{path} already exists and could belong to a running service. Verify ownership/use or choose an isolated node shm_dir. Doctor has not opened, resized, or removed it; allocated cache bytes are only a possible reuse allowance, not confirmed capacity."))
    return results


def _reusable_cache_bytes(directory: Path) -> int | None:
    """Return an optimistic reuse allowance from existing physical allocations.

    This only downgrades apparent capacity failures to warnings, never to a pass.
    Sparse logical lengths, foreign filesystems and duplicate inodes cannot be
    credited as extra free bytes. None means metadata inspection was incomplete.
    """
    try:
        device = directory.stat().st_dev
        allocated = 0
        seen = set()
        for name in ("aegaeon_cpu_cache", "aegaeon_model_cache"):
            path = directory / name
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(metadata.st_mode) or not os.access(path, os.R_OK | os.W_OK):
                continue
            if metadata.st_dev != device or (metadata.st_dev, metadata.st_ino) in seen:
                continue
            seen.add((metadata.st_dev, metadata.st_ino))
            allocated += max(0, min(metadata.st_size, metadata.st_blocks * 512))
        return allocated
    except (OSError, AttributeError, ValueError):
        return None


def _local_memory_available() -> tuple[int, str]:
    content = Path("/proc/meminfo").read_text()
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", content, re.MULTILINE)
    if match is None:
        raise RuntimeError("/proc/meminfo has no MemAvailable field")
    available = int(match.group(1)) * 1024
    detail = f"host MemAvailable={available} bytes"
    # For cgroup v2, inspect the process's own group, or the mounted root if
    # the process path is not visible in the current cgroup namespace.
    # Parent groups can impose additional constraints; this is not a reservation.
    try:
        groups = Path("/proc/self/cgroup").read_text().splitlines()
        group = next(line.split(":", 2)[2] for line in groups if line.startswith("0::"))
        root = Path("/sys/fs/cgroup")
        group_path = root.joinpath(group.lstrip("/")).resolve()
        if not group_path.is_relative_to(root):
            raise RuntimeError("cgroup path is outside the mounted hierarchy")
        cgroup = group_path if (group_path / "memory.max").exists() else root
        limit = (cgroup / "memory.max").read_text().strip()
        if limit == "max":
            detail += "; inspected cgroup v2 memory.max is unlimited"
        else:
            current = int((cgroup / "memory.current").read_text().strip())
            limit = int(limit)
            if min(current, limit) < 0:
                raise ValueError("negative cgroup memory usage or limit")
            remaining = max(0, limit - current)
            available = min(available, remaining)
            detail += f"; inspected cgroup v2 has {remaining} bytes remaining (max={limit}, current={current})"
        detail += "; ancestor cgroup limits are not exhaustively checked"
    except (OSError, ValueError, RuntimeError, StopIteration) as exc:
        detail += f"; cgroup limit unavailable ({type(exc).__name__})"
    return available, detail


def _check_host_memory(budget: int | None, reusable_bytes: int | None = 0) -> CheckResult:
    try:
        available, detail = _local_memory_available()
        if budget is None:
            return _warning("host memory capacity", f"Local {detail}; no configuration budget provided, so capacity is unverified.")
        if available < budget:
            detail = f"Local {detail}; available memory is below the {budget}-byte configured CPU slab plus model cache budget. "
            if reusable_bytes is None:
                return _warning("host memory capacity", detail + "Existing cache allocation could not be inspected reliably; additional memory needed after reuse is unverified.")
            additional = max(0, budget - reusable_bytes)
            if additional <= available:
                return _warning("host memory capacity", detail + f"Existing cache files have {reusable_bytes} allocated bytes that might already be charged to host/cgroup memory, leaving a minimum of {additional} additional bytes. Cache reuse and memory accounting are unverified; full-budget capacity is not confirmed.")
            return CheckResult("host memory capacity", False, detail + f"Even allowing all {reusable_bytes} existing allocated cache bytes for reuse, at least {additional} additional bytes are needed. Reduce cache sizes or make more host/container memory available.")
        return CheckResult("host memory capacity", True, f"Local {detail}; sufficient for the {budget}-byte cache budget at this instant. Worker overhead, concurrent allocations and full-budget pinning remain unverified.")
    except (OSError, ValueError, RuntimeError) as exc:
        return _warning("host memory capacity", f"{type(exc).__name__}: {exc}. Cannot inspect local available host memory; configuration capacity remains unverified.")


def _probe_cuda_registration(directory: str, cuda_version: str) -> str:
    """Register exactly one shared-memory page; always unregister before unmapping.

    Executed in a bounded child process so CUDA failure or cleanup failure cannot
    leave a registration in the doctor process or hang the whole inspection.
    """
    match = re.match(r"(\d+)\.\d+", cuda_version)
    if match is None:
        raise RuntimeError("PyTorch CUDA version unavailable; cannot choose the runtime library")
    library = ctypes.CDLL(f"libcudart.so.{match.group(1)}")
    library.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    library.cudaHostRegister.restype = ctypes.c_int
    library.cudaHostUnregister.argtypes = [ctypes.c_void_p]
    library.cudaHostUnregister.restype = ctypes.c_int
    library.cudaGetErrorString.argtypes = [ctypes.c_int]
    library.cudaGetErrorString.restype = ctypes.c_char_p

    def check(status: int, operation: str) -> None:
        if status:
            message = library.cudaGetErrorString(status)
            description = message.decode("utf-8", errors="replace") if message else "unknown CUDA error"
            raise RuntimeError(f"{operation} failed ({status}): {description}")

    with tempfile.TemporaryFile(prefix="aegaeon-doctor-cuda-", dir=directory) as handle:
        handle.truncate(mmap.PAGESIZE)
        with mmap.mmap(handle.fileno(), mmap.PAGESIZE, access=mmap.ACCESS_WRITE) as mapping:
            mapping[0] = 0
            pointer = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mapping)))
            check(library.cudaHostRegister(pointer, mmap.PAGESIZE, 0), "cudaHostRegister")
            check(library.cudaHostUnregister(pointer), "cudaHostUnregister")
    return f"Local cudaHostRegister/cudaHostUnregister succeeded for {mmap.PAGESIZE} bytes in {directory}; this does not validate the entire cache budget"


def _check_cuda_registration(directory: Path, cuda_version: str | None) -> CheckResult:
    try:
        if not cuda_version:
            raise RuntimeError("PyTorch CUDA version unavailable; fix the runtime dependency check first")
        completed = subprocess.run(
            [sys.executable, "-c", "import sys; import torch; from aegaeon.preflight_host import _probe_cuda_registration; "
             "print(_probe_cuda_registration(sys.argv[1], sys.argv[2]))", str(directory), cuda_version],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return CheckResult("CUDA shared-memory registration", True, completed.stdout.strip())
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()[-2000:]
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        detail = f"{type(exc).__name__}: {exc}"
    return CheckResult("CUDA shared-memory registration", False, detail + ". Check the CUDA runtime/driver, shared-memory mount and host pinning resources before starting workers.")


def check_host(*, cuda_version: str | None = None, memory_budget_bytes: int | None = None,
               shm_dir: str | None = None, check_cuda: bool = False) -> list[CheckResult]:
    """Inspect this process's host only; no remote Ray node is contacted."""
    if memory_budget_bytes is not None and (isinstance(memory_budget_bytes, bool) or not isinstance(memory_budget_bytes, int) or memory_budget_bytes < 0):
        raise ValueError("memory_budget_bytes must be a nonnegative integer or None")
    directory = Path(shm_dir if shm_dir is not None else os.environ.get("AEGAEON_SHM_DIR", "/dev/shm")).resolve()
    results = _check_toolchain(cuda_version)
    reusable_bytes = _reusable_cache_bytes(directory)
    shared_memory = _check_shared_memory(directory, memory_budget_bytes, reusable_bytes)
    results.extend(shared_memory)
    if shared_memory[0].ok or shared_memory[0].severity == "warning":
        results.extend(_check_cache_files(directory))
    results.append(_check_host_memory(memory_budget_bytes, reusable_bytes))
    results.append(_check_memlock(memory_budget_bytes))
    if check_cuda:
        if shared_memory[0].ok:
            results.append(_check_cuda_registration(directory, cuda_version))
        else:
            results.append(CheckResult("CUDA shared-memory registration", False, "Not attempted because the local shared-memory mmap probe did not succeed.", severity=shared_memory[0].severity))
    return results
