"""Download a pinned ComfyUI H3 T2V stack; resume and verify every weight."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path('/srv/models/Comfy-Org/MiniMax-H3-FL2VA-quant')
REPO = 'Comfy-Org/MiniMax-H3'
REVISION = 'e5eb578a89295337b8ff433a035929ce0279e0b6'
FILES = [
    ('diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors', 20958205608, '12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993'),
    ('text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 15687142551, '35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6'),
    ('vae/minimax_h3_video_vae_fp16.safetensors', 5207808496, '7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522'),
    ('vae/minimax_h3_audio_vae_fp32.safetensors', 605254808, '8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48'),
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download_one(item):
    relative, size, sha = item
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size and not Path(str(path) + '.aria2').exists():
        if digest(path) == sha:
            print('Already verified:', relative, flush=True)
            return
        raise RuntimeError('Existing complete file has wrong SHA256: ' + str(path))
    for attempt in range(5):
        # Cache-busting refreshes expiring Xet signed redirects when retrying.
        url = 'https://hf-mirror.com/{}/resolve/{}/{}?download=true&t={}'.format(REPO, REVISION, relative, int(time.time()))
        print('Downloading:', relative, 'attempt', attempt + 1, flush=True)
        ms_url = 'https://modelscope.cn/models/{}/resolve/master/{}'.format(REPO, relative)
        result = subprocess.run(['aria2c', '--continue=true', '--auto-file-renaming=false',
            '--allow-overwrite=false', '--file-allocation=none', '--max-connection-per-server=8',
            '--split=16', '--min-split-size=16M', '--connect-timeout=20', '--timeout=60',
            '--max-tries=5', '--retry-wait=5', '--summary-interval=30', '--console-log-level=warn',
            '--dir=' + str(path.parent), '--out=' + path.name, ms_url, url])
        if result.returncode == 0:
            break
    if result.returncode != 0:
        raise RuntimeError('Download failed: ' + relative)
    if path.stat().st_size != size or digest(path) != sha:
        raise RuntimeError('Integrity check failed: ' + relative)
    path.chmod(0o644)
    print('Verified SHA256:', relative, flush=True)


def main():
    os.umask(0o022)
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = dict(repo=REPO, revision=REVISION, endpoints=['https://modelscope.cn', 'https://hf-mirror.com'],
                    status='downloading', files=[dict(path=p, size=s, sha256=h) for p, s, h in FILES])
    (ROOT / 'download-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download_one, FILES))
    # README is provenance/help, not a required model component.
    if not (ROOT / 'UPSTREAM_README.md').exists():
        subprocess.run(['curl', '-fL', '--max-time', '60', '-sS',
            'https://modelscope.cn/models/{}/resolve/master/README.md'.format(REPO),
            '-o', str(ROOT / 'UPSTREAM_README.md')], check=True)
    manifest['status'] = 'complete'
    manifest['verified_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (ROOT / 'download-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('ALL FILES VERIFIED', flush=True)


if __name__ == '__main__':
    main()
