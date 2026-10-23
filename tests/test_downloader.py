import hashlib
import requests
import pytest
import downloader

@pytest.fixture(autouse=True)
def use_http_fallback(monkeypatch):
    monkeypatch.setattr(downloader.shutil, 'which', lambda _: None)

class Response:
    def __init__(self, content, status=200, headers=None):
        self.content, self.status_code, self.headers = content, status, headers or {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def raise_for_status(self):
        pass
    def iter_content(self, size):
        yield self.content

@pytest.mark.parametrize('resume', [False, True])
def test_verified_download(tmp_path, monkeypatch, resume):
    target = tmp_path / 'weights.safetensors'
    partial = tmp_path / 'weights.safetensors.part'
    content = b'valid model weights'
    if resume:
        partial.write_bytes(content[:5])
    def get(url, headers, **kwargs):
        assert headers == ({'Range': 'bytes=5-'} if resume else {})
        return Response(content[5:] if resume else content, 206 if resume else 200,
                        {'Content-Range': f'bytes 5-{len(content)-1}/{len(content)}'})
    monkeypatch.setattr(downloader.requests, 'get', get)
    downloader.fetch_file('https://example.test/weights', target, len(content), hashlib.sha256(content).hexdigest())
    assert target.read_bytes() == content
    assert not partial.exists()


def test_interrupted_transfer_resumes(tmp_path, monkeypatch):
    content = b'valid weights'
    calls = []
    class Interrupted(Response):
        def iter_content(self, size):
            yield content[:5]
            raise requests.ConnectionError('connection lost')
    def get(url, headers, **kwargs):
        calls.append(headers)
        return Interrupted(b'') if len(calls) == 1 else Response(content[5:], 206, {'Content-Range': 'bytes 5-12/13'})
    monkeypatch.setattr(downloader.requests, 'get', get)
    monkeypatch.setattr(downloader.time, 'sleep', lambda _: None)
    target = tmp_path / 'weights'
    downloader.fetch_file('https://example.test/model', target, len(content), hashlib.sha256(content).hexdigest())
    assert calls == [{}, {'Range': 'bytes=5-'}]
    assert target.read_bytes() == content


def test_external_aria2_file_is_left_alone(tmp_path, monkeypatch):
    partial = tmp_path / 'weights.aria2dl'
    partial.write_bytes(b'already')
    (tmp_path / 'weights.aria2dl.aria2').write_bytes(b'control')

    def fail(*args, **kwargs):
        raise AssertionError('不应再下载已有的 aria2 文件')

    monkeypatch.setattr(downloader, 'fetch_file', fail)
    monkeypatch.setattr(downloader, 'hf_hub_download', fail)
    spec = {'repo': 'example', 'revision': 'abc', 'files': {'weights': 10},
            'sha256': {'weights': 'abc'}}
    downloader.download_model(spec, tmp_path)
    assert partial.read_bytes() == b'already'


def test_corrupt_download_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader.requests, 'get', lambda *a, **kw: Response(b'bad'))
    monkeypatch.setattr(downloader.time, 'sleep', lambda _: None)
    target = tmp_path / 'weights'
    with pytest.raises(RuntimeError, match='SHA-256'):
        downloader.fetch_file('https://example.test/model', target, 3, hashlib.sha256(b'yes').hexdigest())
    assert not target.exists()
