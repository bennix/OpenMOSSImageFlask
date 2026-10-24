import time
import pytest
import app as module

@pytest.fixture
def client():
    return module.app.test_client()

@pytest.mark.parametrize('payload', [None, [], {}, {'prompt': ' '}, {'prompt': 'cat', 'width': 257},
    {'prompt': 'cat', 'steps': 0}, {'prompt': 'cat', 'seed': -1},
    {'prompt': 'cat', 'width': True}, {'prompt': 'cat', 'negative': 1},
    {'prompt': 'cat', 'negative': 'x' * 12001}, {'prompt': 'cat', 'enhance': 'false'},
    {'prompt': 'cat', 'enhance': 'poster'}, {'prompt': 'cat', 'region': {'x': -1, 'y': 0, 'w': 10, 'h': 10, 'source_width': 512, 'source_height': 512, 'note': '石头'}},
    {'prompt': 'cat', 'region': {'x': 1, 'y': 1, 'w': 20, 'h': 20, 'source_width': 512, 'source_height': 512, 'note': '  ', 'kind': 'text'}},
    {'prompt': 'cat', 'region': {'x': 1, 'y': 1, 'w': 20, 'h': 20, 'source_width': 512, 'source_height': 512, 'note': '石头', 'kind': 'poster'}},
    {'prompt': 'cat', 'lora': 'yes'}, {'prompt': 'cat', 'lora_scale': 3}])
def test_invalid_generation(client, payload):
    response = client.post('/api/generate', json=payload)
    assert response.status_code == 400


def test_lora_defaults_and_weight_gate(client, monkeypatch):
    plain = module.validate({'prompt': 'cat'})
    assert plain['lora'] is False and plain['lora_scale'] == 0.7 and plain['negative'] == '' and plain['region'] is None
    assert module.validate({'prompt': 'cat', 'enhance': 'layout'})['enhance'] == 'layout'
    boxed = module.validate({'prompt': 'cat', 'region': {'x': 10, 'y': 20, 'w': 30, 'h': 40, 'source_width': 512, 'source_height': 512, 'note': '  石头  '}})
    assert boxed['region']['note'] == '石头' and boxed['region']['w'] == 30 and boxed['region']['kind'] == 'edit'
    assert module.validate({'prompt': 'cat', 'region': {'x': 1, 'y': 1, 'w': 20, 'h': 20, 'source_width': 512, 'source_height': 512, 'note': '青苔', 'kind': 'text'}})['region']['kind'] == 'text'
    assert module.validate({'prompt': 'cat', 'negative': '  模糊  '})['negative'] == '模糊'
    assert module.validate({'prompt': 'cat', 'lora': True, 'lora_scale': 0.7})['lora'] is True
    monkeypatch.setattr(module, 'model_status', lambda key: {'ready': key != 'lora', 'repo': key})
    refused = client.post('/api/generate', json={'prompt': 'cat', 'lora': True})
    assert refused.status_code == 409
    assert 'Moss-image-nsfw-uncensored-lora' in refused.json['error']


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
    page = client.get('/').data
    assert b'NSFW' not in page and b'nsfw' not in page and b'moss_xvstyle' not in page
    assert b'id="lora"' not in page
    assert b'id="upscale"' in client.get('/').data
    assert '高清'.encode() in client.get('/').data
    assert b'id="negative"' in client.get('/').data
    assert '负向提示词'.encode() in client.get('/').data
    assert '框选'.encode() in client.get('/').data
    assert '添加文字'.encode() in client.get('/').data
    assert '后期修改'.encode() in client.get('/').data
    assert '修改这块'.encode() in client.get('/').data
    assert b'id="ratio"' in client.get('/').data


def test_upscale_validation(client, monkeypatch):
    assert client.post('/api/upscale', json={'src': '/etc/passwd', 'mode': 'fast'}).status_code == 400
    assert client.post('/api/upscale', json={'src': '/about/forest-cat.png', 'mode': 'other'}).status_code == 400
    monkeypatch.setattr(module, 'model_status', lambda key: {'ready': False, 'repo': 'xinntao/Real-ESRGAN'})
    missing = client.post('/api/upscale', json={'src': '/about/forest-cat.png', 'mode': 'fast'})
    assert missing.status_code == 409
    assert 'Real-ESRGAN' in missing.json['error']
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
