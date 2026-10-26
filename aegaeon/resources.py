"""Locate installed web assets and deployment-specific latency profiles."""

import os
from pathlib import Path


def web_directory() -> Path:
    configured = os.environ.get("AEGAEON_WEB_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    package = Path(__file__).resolve().parent
    bundled = package / "_web"
    return bundled if bundled.is_dir() else package.parent / "web"


def profile_root() -> Path:
    configured = os.environ.get("AEGAEON_PROFILE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    package = Path(__file__).resolve().parent
    bundled = package / "_profiles"
    return bundled if bundled.is_dir() else package.parent / "profiles"
