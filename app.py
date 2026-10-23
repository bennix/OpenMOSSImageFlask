"""Local Flask UI; heavy model work runs in a disposable subprocess."""
import json
import os
import struct
import time
from pathlib import Path
import subprocess
import sys
import threading
import uuid

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory

ROOT = Path(__file__).resolve().parent
MODELS = json.loads((ROOT / 'models.json').read_text())
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True
jobs = {}
lock = threading.Lock()
samples = {}
sample_lock = threading.Lock()
ARIA2_BLOCK = 16 * 1024


def track_speed(key, nbytes):
    """Average over the last 12 seconds so 1 MB chunks still show a speed."""
    now = time.monotonic()
    with sample_lock:
        history = samples.setdefault(key, [])
        history.append((now, nbytes))
        cutoff = now - 12
        while len(history) > 2 and history[0][0] < cutoff:
            history.pop(0)
        if now - history[0][0] < 0.5:
            return None
        return max(0, round((nbytes - history[0][1]) / (now - history[0][0])))


def _take(data, offset, fmt):
    size = struct.calcsize(fmt)
    if offset + size > len(data):
        raise ValueError('aria2 控制文件不完整')
    return struct.unpack_from(fmt, data, offset)[0], offset + size


def _bits(field, length, block):
    count = (length + block - 1) // block
    got = 0
    for index in range(count):
        if index // 8 >= len(field) or not field[index // 8] & (0x80 >> (index % 8)):
            continue
        got += block if index < count - 1 else length - block * (count - 1)
    return got


def aria2_received(control, expected):
    """Bytes finished in an aria2 control file, including in-flight blocks."""
    data = control.read_bytes()
    offset = 0
    version, offset = _take(data, offset, '>H')
    if version != 1:
        raise ValueError('不认识的 aria2 控制文件')
    _, offset = _take(data, offset, '>I')
    hash_length, offset = _take(data, offset, '>I')
    offset += hash_length
    piece, offset = _take(data, offset, '>I')
    total, offset = _take(data, offset, '>Q')
    _, offset = _take(data, offset, '>Q')
    bitfield_length, offset = _take(data, offset, '>I')
    if piece <= 0 or offset + bitfield_length > len(data):
        raise ValueError('aria2 控制文件不完整')
    bitfield = data[offset:offset + bitfield_length]
    offset += bitfield_length
    got = _bits(bitfield, total, piece)
    inflight, offset = _take(data, offset, '>I')
    for _ in range(inflight):
        _, offset = _take(data, offset, '>I')
        length, offset = _take(data, offset, '>I')
        field_length, offset = _take(data, offset, '>I')
        if offset + field_length > len(data):
            raise ValueError('aria2 控制文件不完整')
        got += _bits(data[offset:offset + field_length], length, ARIA2_BLOCK)
        offset += field_length
    return min(got, expected, total)


def received_bytes(target, expected):
    if target.is_file() and target.stat().st_size == expected:
        return expected
    best = 0
    for partial in (target.with_suffix(target.suffix + '.part'), Path(str(target) + '.aria2dl')):
        if not partial.is_file():
            continue
        control = Path(str(partial) + '.aria2')
        if control.is_file():
            try:
                best = max(best, aria2_received(control, expected))
                continue
            except (OSError, ValueError, struct.error):
                pass
        stat = partial.stat()
        best = max(best, min(stat.st_size, stat.st_blocks * 512, expected))
    return best


def model_status(key):
    spec = MODELS[key]
    folder = ROOT / 'models' / spec['folder']
    total = sum(spec['files'].values())
    files = []
    received = 0
    for name, size in spec['files'].items():
        got = received_bytes(folder / name, size)
        received += got
        files.append({'name': name, 'bytes': got, 'total': size, 'done': got == size,
                      'speed': track_speed(f'{key}:{name}', got)})
    received = min(total, received)
    return {'repo': spec['repo'], 'ready': received == total, 'bytes': received, 'total': total,
            'speed': track_speed(key, received), 'files': files}


def validate(data):
    if not isinstance(data, dict):
        raise ValueError('请求必须是 JSON 对象。')
    prompt = data.get('prompt', '')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
        raise ValueError('请输入 1–12000 字符的提示词。')
    result = {'prompt': prompt.strip()}
    for field, default, low, high in [('width', 512, 256, 1024), ('height', 512, 256, 1024),
                                      ('steps', 30, 1, 75), ('seed', 42, 0, 2**32 - 1)]:
        value = data.get(field, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{field} 必须是 {low}–{high} 之间的整数。')
        if field in ('width', 'height') and value % 16:
            raise ValueError('宽高必须是 16 的倍数。')
        result[field] = value
    enhance = data.get('enhance', False)
    if type(enhance) is not bool:
        raise ValueError('enhance 必须是布尔值。')
    result['enhance'] = enhance
    return result


def run_job(job_id, action, payload):
    directory = ROOT / 'outputs' / job_id
    directory.mkdir(parents=True)
    (directory / 'request.json').write_text(json.dumps(payload, ensure_ascii=False))
    try:
        with (directory / 'worker.log').open('w') as log:
            process = subprocess.run([sys.executable, str(ROOT / 'worker.py'), action, str(directory)],
                                     cwd=ROOT, stdout=log, stderr=log)
        if process.returncode:
            raise RuntimeError(f'模型进程退出（{process.returncode}）。详见 outputs/{job_id}/worker.log')
        result = json.loads((directory / 'result.json').read_text())
        with lock:
            jobs[job_id].update(state='done', result=result)
    except Exception as exc:
        with lock:
            jobs[job_id].update(state='error', error=str(exc))


def submit(action, payload):
    with lock:
        if any(j['state'] == 'running' for j in jobs.values()):
            return jsonify(error='已有下载或生成任务正在运行。'), 409
        job_id = uuid.uuid4().hex
        jobs[job_id] = {'id': job_id, 'state': 'running', 'action': action}
    threading.Thread(target=run_job, args=(job_id, action, payload), daemon=True).start()
    return jsonify(id=job_id), 202


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/about')
def about():
    return redirect('/about/', code=302)


@app.get('/about/')
def about_page():
    return send_from_directory(ROOT / 'docs', 'index.html')


@app.get('/about/<path:name>')
def about_file(name):
    return send_from_directory(ROOT / 'docs', name)


@app.get('/api/status')
def status():
    with lock:
        active = next((j['id'] for j in jobs.values() if j['state'] == 'running'), None)
    return jsonify(models={key: model_status(key) for key in MODELS}, active=active)


@app.post('/api/download')
def download():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or data.get('model') not in MODELS:
        return jsonify(error='model 必须为 moss 或 prompt。'), 400
    return submit('download', {'model': data['model']})


@app.post('/api/generate')
def generate():
    try:
        data = validate(request.get_json(silent=True))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    for key in (['moss', 'prompt'] if data['enhance'] else ['moss']):
        if not model_status(key)['ready']:
            return jsonify(error=f'请先下载 {MODELS[key]["repo"]}。'), 409
    return submit('generate', data)


@app.get('/api/jobs/<job_id>')
def job(job_id):
    with lock:
        value = dict(jobs.get(job_id, {}))
    if not value:
        return jsonify(error='任务不存在；服务重启后请查看 outputs 目录。'), 404
    progress = ROOT / 'outputs' / job_id / 'progress.json'
    if progress.exists():
        value['progress'] = json.loads(progress.read_text())
    return jsonify(value)


@app.get('/outputs/<job_id>/image.png')
def image(job_id):
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        return jsonify(error='无效图片路径。'), 404
    return send_from_directory(ROOT / 'outputs' / job_id, 'image.png')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', '7860')), threaded=True, debug=False)
