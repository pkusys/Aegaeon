"""Retag the known NVIDIA cuSPARSELt 0.8.0 ARM64 wheel without changing its libraries."""

import argparse
from email import message_from_bytes
import hashlib
from pathlib import Path


UPSTREAM_FILENAME = "nvidia_cusparselt_cu13-0.8.0-py3-none-manylinux2014_aarch64.whl"
UPSTREAM_SHA256 = "400c6ed1cf6780fc6efedd64ec9f1345871767e6a1a0a552a1ea0578117ea77c"
BUILD = "1aegaeonfix"
FIXED_FILENAME = f"nvidia_cusparselt_cu13-0.8.0-{BUILD}-py3-none-manylinux2014_aarch64.whl"
DIST_INFO = "nvidia_cusparselt_cu13-0.8.0.dist-info"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def repair_wheel(source, output_dir):
    from wheel.wheelfile import WheelFile

    source = Path(source)
    if source.name != UPSTREAM_FILENAME or sha256(source) != UPSTREAM_SHA256:
        raise ValueError("Expected the unmodified NVIDIA cuSPARSELt 0.8.0 aarch64 wheel; filename/SHA256 mismatch")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / FIXED_FILENAME
    # Exclusive creation keeps an existing repaired artifact intact.
    with target.open("xb"):
        pass
    try:
        with WheelFile(str(source)) as original, WheelFile(str(target), mode="w") as repaired:
            for entry in original.infolist():
                if entry.filename == DIST_INFO + "/RECORD":
                    continue
                content = original.read(entry.filename)
                if entry.filename == DIST_INFO + "/WHEEL":
                    info = message_from_bytes(content)
                    if info.get_all("Tag") != ["py3-none-manylinux2014_sbsa"]:
                        raise ValueError("Unexpected upstream WHEEL tags")
                    info.replace_header("Tag", "py3-none-manylinux2014_aarch64")
                    info["Build"] = BUILD
                    content = info.as_bytes()
                repaired.writestr(entry, content)
    except BaseException:
        target.unlink()
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    target = repair_wheel(args.source, args.output_dir)
    print(f"{sha256(target)}  {target}")


if __name__ == "__main__":
    main()
