import time
import pytest
import app as module

@pytest.fixture
def client():
    return module.app.test_client()

@pytest.mark.parametrize('payload', [None, [], {}, {'prompt': ' '}, {'prompt': 'cat', 'width': 257},
    {'prompt': 'cat', 'steps': 0}, {'prompt': 'cat', 'seed': -1},
    {'prompt': 'cat', 'width': True}, {'prompt': 'cat', 'enhance': 'false'}])
def test_invalid_generation(client, payload):
    response = client.post('/api/generate', json=payload)
    assert response.status_code == 400


def test_missing_weights(client, monkeypatch):
    monkeypatch.setattr(module, 'model_status', lambda key: {'ready': False})
    assert client.post('/api/generate', json={'prompt': 'cat'}).status_code == 409


def test_job_lifecycle(client, monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'jobs', {})
    monkeypatch.setattr(module, 'model_status', lambda key: {'ready': True})
    def worker(args, **kwargs):
        folder = tmp_path / 'outputs' / args[-1].split('/')[-1]
        (folder / 'result.json').write_text('{"image":"/example.png"}')
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(module.subprocess, 'run', worker)
    response = client.post('/api/generate', json={'prompt': 'cat'})
    assert response.status_code == 202
    for _ in range(100):
        result = client.get('/api/jobs/' + response.json['id']).json
        if result['state'] != 'running':
            break
        time.sleep(.01)
    assert result['state'] == 'done'
    assert result['result']['image'] == '/example.png'


def test_busy(client, monkeypatch):
    monkeypatch.setattr(module, 'jobs', {'busy': {'state': 'running'}})
    assert client.post('/api/download', json={'model': 'moss'}).status_code == 409


def test_paths_and_download_validation(client):
    assert client.get('/').status_code == 200
    assert b'fmtSpeed' in client.get('/').data
    assert b'id="gen"' in client.get('/').data
    assert b'id="viewport"' in client.get('/').data
    assert '复制'.encode() in client.get('/').data
    assert '保存'.encode() in client.get('/').data
    about = client.get('/about/')
    assert about.status_code == 200
    assert '邱锡鹏'.encode() in about.data
    assert '计算与智能创新学院'.encode() in about.data
    assert '水彩插画风格'.encode() in about.data
    assert client.get('/about').status_code == 302
    image = client.get('/about/forest-cat.png')
    assert image.status_code == 200
    assert image.mimetype == 'image/png'
    assert client.get('/outputs/invalid/image.png').status_code == 404
    assert client.get('/api/jobs/unknown').status_code == 404
    assert client.post('/api/download', json={'model': '../other'}).status_code == 400


def test_download_progress_speed_and_aria2(monkeypatch, tmp_path):
    import struct
    folder = tmp_path / 'models' / 'demo'
    folder.mkdir(parents=True)
    (folder / 'done.bin').write_bytes(b'1234')
    (folder / 'weights.bin.part').write_bytes(b'x' * 40)
    control = struct.pack('>HIIIQQI', 1, 0, 0, 50, 100, 0, 1) + bytes([0x80]) + struct.pack('>I', 0)
    sparse = folder / 'sparse.bin.aria2dl'
    sparse.write_bytes(b'')
    os_truncate = __import__('os').truncate
    os_truncate(sparse, 100000)
    (folder / 'sparse.bin.aria2dl.aria2').write_bytes(control)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'MODELS', {'demo': {'folder': 'demo', 'repo': 'example', 'files': {
        'done.bin': 4, 'weights.bin': 100, 'sparse.bin': 100}}})
    monkeypatch.setattr(module, 'samples', {})
    first = module.model_status('demo')
    assert first['bytes'] == 94
    assert first['speed'] is None
    assert [(item['name'], item['bytes'], item['done']) for item in first['files']] == [
        ('done.bin', 4, True), ('weights.bin', 40, False), ('sparse.bin', 50, False)]
    time.sleep(0.6)
    (folder / 'weights.bin.part').write_bytes(b'y' * 80)
    second = module.model_status('demo')
    assert second['files'][1]['bytes'] == 80
    assert second['files'][1]['speed'] > 0
    assert second['speed'] > 0
