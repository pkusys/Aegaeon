import os

from aegaeon.logger import init_logger

logger = init_logger(__name__)

_HF = "huggingface"
_MS = "modelscope"

_MS_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")
_HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def download_model(name: str) -> str:
    repo = _get_repo_type()
    return _snapshot(repo, name)


def _get_repo_type() -> str:
    return os.environ.get("AEGAEON_DEFAULT_MODEL_REPO", _MS)


def _snapshot(repo: str, name: str) -> str:
    cache_dir = _MS_CACHE if repo == _MS else _HF_CACHE
    local = os.path.join(cache_dir, name)
    if os.path.isdir(local):
        logger.debug("Model %s found in local cache", name)
        return local

    logger.info("Downloading %s via %s ...", name, repo)
    if repo == _MS:
        from modelscope.hub import snapshot_download
        return snapshot_download(name)
    elif repo == _HF:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=name)
    else:
        raise ValueError(f"Unknown AEGAEON_DEFAULT_MODEL_REPO value: {repo}")
