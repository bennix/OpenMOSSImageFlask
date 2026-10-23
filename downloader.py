"""Download pinned public model files with range resume and SHA-256 checks."""
import hashlib
import shutil
import subprocess
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from huggingface_hub import hf_hub_download, hf_hub_url


def fetch_file(url, target, expected_size, sha256):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    if shutil.which('aria2c'):
        subprocess.run([
            'aria2c', '--no-conf', '--split=16', '--max-connection-per-server=16',
            '--min-split-size=8M', '--file-allocation=none', '--continue=true',
            '--auto-file-renaming=false', '--allow-overwrite=true', '--max-tries=10',
            '--retry-wait=5', '--connect-timeout=20', '--timeout=30',
            '--summary-interval=30', '--console-log-level=warn',
            f'--checksum=sha-256={sha256}', f'--dir={target.parent}',
            f'--out={partial.name}', url,
        ], check=True)
        if partial.stat().st_size != expected_size:
            raise RuntimeError(f'下载长度不符：{target.name}')
        partial.replace(target)
        return
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected_size:
                partial.unlink()
                offset = 0
            if offset < expected_size:
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                with requests.get(url, headers=headers, stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    if offset and response.status_code == 206:
                        if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                            raise RuntimeError('服务器返回不匹配的续传范围。')
                    else:
                        offset = 0
                    with partial.open('ab' if offset else 'wb') as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
            if partial.stat().st_size != expected_size:
                raise RuntimeError(f'下载长度不符：{target.name}')
            with partial.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            if checksum != sha256:
                partial.unlink()
                raise RuntimeError(f'SHA-256 校验失败：{target.name}')
            partial.replace(target)
            return
        except (requests.RequestException, RuntimeError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def download_model(spec, folder):
    folder = Path(folder)
    def download(name):
        target = folder / name
        size = spec['files'][name]
        if target.is_file() and target.stat().st_size == size:
            return
        external = Path(str(target) + '.aria2dl')
        if external.is_file() and (external.stat().st_size > 0 or Path(str(external) + '.aria2').is_file()):
            return
        checksum = spec.get('sha256', {}).get(name)
        if checksum:
            url = spec.get('mirrors', {}).get(name) or hf_hub_url(spec['repo'], name, revision=spec['revision'])
            fetch_file(url, target, size, checksum)
        else:
            hf_hub_download(spec['repo'], name, revision=spec['revision'], local_dir=folder)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, spec['files']))
