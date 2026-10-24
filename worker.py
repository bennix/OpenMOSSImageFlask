"""One process per job keeps MLX and PyTorch weights out of the Flask server."""
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
MODELS = json.loads((ROOT / 'models.json').read_text())
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')


def report(directory, stage, **details):
    temporary = directory / 'progress.tmp'
    temporary.write_text(json.dumps({'stage': stage, **details}, ensure_ascii=False))
    temporary.replace(directory / 'progress.json')
    print(stage, details, flush=True)


UPSCALE_MODES = {
    'photo': ('RealESRGAN_x4plus.pth', 23, 4, '照片 4×'),
    'anime': ('RealESRGAN_x4plus_anime_6B.pth', 6, 4, '插画 4×'),
    'fast': ('RealESRGAN_x2plus.pth', 23, 2, '快速 2×'),
}


def upscale_request(src, mode):
    if src == '/about/forest-cat.png':
        rel = 'docs/forest-cat.png'
    elif isinstance(src, str) and src.startswith('/outputs/') and src.endswith('/image.png'):
        rel = src[1:]
        job_id = rel[len('outputs/'):-len('/image.png')]
        if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            raise ValueError('只能放大当前工作台里的图片。')
    else:
        raise ValueError('只能放大当前工作台里的图片。')
    if mode not in UPSCALE_MODES:
        raise ValueError('mode 必须是 photo、anime 或 fast。')
    if not (ROOT / rel).is_file():
        raise ValueError('图片不存在。')
    return {'path': rel, 'mode': mode}


def download(directory, data):
    from downloader import download_model
    spec = MODELS[data['model']]
    report(directory, '下载权重（支持断点续传）')
    download_model(spec, ROOT / 'models' / spec['folder'])
    return {'model': data['model']}


def fuse_lora(transformer, folder, scale):
    """Add scale * (alpha / r) * B @ A into matching DiT linear weights."""
    from safetensors.torch import load_file
    config = json.loads((folder / 'adapter_config.json').read_text())
    factor = float(scale) * config['lora_alpha'] / config['r']
    paired = {}
    for key, tensor in load_file(folder / 'pytorch_lora_weights.safetensors').items():
        name = key.removeprefix('transformer.')
        kind = 'A' if '.lora_A.' in name else 'B' if '.lora_B.' in name else ''
        if not kind:
            raise RuntimeError(f'无法识别的 LoRA 权重：{key}')
        paired.setdefault(name.split(f'.lora_{kind}.')[0], {})[kind] = tensor
    modules = dict(transformer.named_modules())
    for name, pair in paired.items():
        layer = modules.get(name)
        if layer is None or 'A' not in pair or 'B' not in pair:
            raise RuntimeError(f'LoRA 层对不上：{name}')
        weight = layer.weight.data
        delta = pair['B'].to(dtype=weight.dtype, device=weight.device) @ pair['A'].to(dtype=weight.dtype, device=weight.device)
        if delta.shape != weight.shape:
            raise RuntimeError(f'LoRA 形状不符：{name}')
        weight.add_(delta * factor)
    return len(paired)


def _json_object(text):
    raw = text.strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0]
    start, end = raw.find('{'), raw.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('提示词模型没有返回 JSON。')
    return json.loads(raw[start:end + 1])


def enhance_prompt(data):
    from mlx_lm import load, generate
    from vendor.compile_prompt import compile_prompt, region_item, region_prompt, scale_box
    lora = data.get('lora', False)
    mode = data['enhance']
    region = data.get('region')
    box = scale_box(region, data['width'], data['height']) if region else None
    limit = ('Keep moss_xvstyle at the start and retain adult content the user wrote.'
             if lora else 'Do not add sexual or explicit content.')
    if mode == 'layout':
        system = ('Return one JSON Layout draft and nothing else. Use width and height exactly as given. '
                  'bbox_coord_space must be "abs" and bbox_format "xyxy". Boxes are [x_min,y_min,x_max,y_max] '
                  'pixels from the top-left. summary.caption describes the whole image. items is a nonempty array; '
                  'each item has label (object, character, or text), caption, and bbox. Text items also have text. '
                  + ('One item must use the supplied user box exactly. ' if box else '') + limit)
    else:
        system = ('Return exactly {"caption":"..."} . The caption is one detailed English image description. '
                  'Preserve quoted text verbatim. Do not include coordinates. ' + limit)
    user = data['prompt']
    if data.get('negative'):
        user += '\nAvoid: ' + data['negative']
    if box:
        user += (f"\nCanvas {data['width']}x{data['height']}. User box [x1,y1,x2,y2]={box}. "
                 f"Edit note: {region.get('note') or 'revise this region.'}")
    model, tokenizer = load(str(ROOT / 'models' / 'prompt'))
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    result = generate(model, tokenizer, prompt=rendered, max_tokens=900 if mode == 'layout' else 384, verbose=False)
    if not result.strip():
        raise RuntimeError('提示词模型返回空内容。')
    if mode == 'caption' and not box:
        try:
            return _json_object(result)['caption'].strip()
        except (ValueError, KeyError, json.JSONDecodeError):
            if '{' in result:
                raise RuntimeError('提示词模型没有返回可用的描述。')
            return result.strip()
    if mode == 'caption':
        try:
            summary = _json_object(result)['caption'].strip()
        except (ValueError, KeyError, json.JSONDecodeError):
            summary = result.strip()
        return region_prompt(summary, region.get('note') or summary, box, data['width'], data['height'], region.get('kind', 'edit'))
    draft = _json_object(result)
    draft['width'], draft['height'] = data['width'], data['height']
    draft['bbox_coord_space'], draft['bbox_format'] = 'abs', 'xyxy'
    if box:
        items = draft.get('items') if isinstance(draft.get('items'), list) else []
        note = region.get('note') or data['prompt']
        placed = region_item(region.get('kind', 'edit'), note, box)
        rest = [entry for entry in items[1:] if isinstance(entry, dict)] if items else []
        draft['items'] = [placed, *rest]
    try:
        return compile_prompt(draft)
    except ValueError as exc:
        if box:
            return region_prompt(data['prompt'], region.get('note') or data['prompt'], box, data['width'], data['height'], region.get('kind', 'edit'))
        raise RuntimeError(f'Layout 无法编译：{exc}') from exc


def generate_image(directory, data):
    prompt = data['prompt']
    if data['enhance'] != 'off':
        # A separate process guarantees all MLX allocations are released before MOSS loads.
        report(directory, 'MLX 提示词扩写')
        import subprocess
        subprocess.run([sys.executable, __file__, 'enhance', str(directory)], check=True)
        prompt = (directory / 'prompt.txt').read_text()
    elif data.get('region'):
        from vendor.compile_prompt import append_region, scale_box
        box = scale_box(data['region'], data['width'], data['height'])
        prompt = append_region(prompt, data['region'].get('note') or prompt, box, data['width'], data['height'], data['region'].get('kind', 'edit'))
    if data.get('lora'):
        trigger = MODELS['lora']['trigger']
        if trigger not in prompt:
            if prompt.startswith('{'):
                layout = json.loads(prompt)
                layout['summary']['caption'] = f'{trigger}, {layout["summary"]["caption"]}'
                prompt = json.dumps(layout, ensure_ascii=False, separators=(',', ':'))
            else:
                prompt = f'{trigger}, {prompt}'
    (directory / 'prompt.txt').write_text(prompt)

    import torch
    from transformers import AutoModel, AutoTokenizer
    from diffusers import AutoencoderKLFlux2
    from vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageTransformer, MOSSImageScheduler
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    model_path = ROOT / 'models' / 'moss-sft'
    report(directory, '加载文本编码器', device=device)
    encoder = AutoModel.from_pretrained(model_path / 'text_encoder', dtype=dtype,
                                        local_files_only=True).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path / 'tokenizer', local_files_only=True)
    pipe = MOSSImagePipeline(transformer=None, scheduler=MOSSImageScheduler(),
                             text_encoder=encoder, tokenizer=tokenizer)
    report(directory, '编码提示词')
    positive, positive_mask = pipe.encode_prompt(prompt, max_sequence_length=512)
    negative_text = data.get('negative', '')
    negative, negative_mask = pipe.encode_prompt(negative_text, max_sequence_length=512)
    pipe.text_encoder = None
    pipe.tokenizer = None
    del encoder, tokenizer
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    elif device == 'cuda':
        torch.cuda.empty_cache()

    report(directory, '加载图像模型')
    transformer = MOSSImageTransformer.from_pretrained(model_path / 'transformer',
                        torch_dtype=dtype, local_files_only=True).eval()
    if data.get('lora'):
        report(directory, '加载 LoRA', scale=data['lora_scale'])
        fuse_lora(transformer, ROOT / 'models' / MODELS['lora']['folder'], data['lora_scale'])
    pipe.transformer = transformer.to(device)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(model_path / 'vae', torch_dtype=torch.float32,
                                                 local_files_only=True).eval().to(device)
    pipe.vae.enable_tiling()
    def progress(items):
        for step, item in enumerate(items):
            report(directory, '图像采样', step=step, total=data['steps'])
            yield item
        report(directory, '解码图片', step=data['steps'], total=data['steps'])
    pipe.progress_bar = progress
    image = pipe(prompt_embeds=positive, prompt_attention_mask=positive_mask,
                 negative_prompt_embeds=negative, negative_prompt_attention_mask=negative_mask,
                 width=data['width'], height=data['height'], num_inference_steps=data['steps'],
                 guidance_scale=3.0, uncond_skip_layers=[] if negative_text else [22],
                 generator=torch.Generator('cpu').manual_seed(data['seed'])).images[0]
    image.save(directory / 'image.png')
    shown = prompt if not negative_text else f'{prompt}\n负向：{negative_text}'
    return {'image': f'/outputs/{directory.name}/image.png', 'prompt': shown, 'device': device,
            'width': data['width'], 'height': data['height'], 'steps': data['steps'], 'seed': data['seed']}


def upscale_image(directory, data):
    import torch
    from PIL import Image
    from vendor.rrdbnet import enhance, load_rrdb
    filename, blocks, scale, label = UPSCALE_MODES[data['mode']]
    source = ROOT / data['path']
    if not source.is_file() or data['path'] != upscale_request(
            '/about/forest-cat.png' if data['path'] == 'docs/forest-cat.png' else '/' + data['path'], data['mode'])['path']:
        raise RuntimeError('无效图片。')
    image = Image.open(source)
    if max(image.size) > 2048:
        raise RuntimeError('图片边长超过 2048，不再继续放大。')
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    report(directory, '加载放大模型', device=device, mode=label)
    model = load_rrdb(ROOT / 'models' / 'upscale' / filename, blocks, scale, device)
    result = enhance(model, image, device, progress=lambda step, total: report(directory, '高清放大', step=step, total=total))
    result.save(directory / 'image.png')
    return {'image': f'/outputs/{directory.name}/image.png',
            'prompt': f'Real-ESRGAN {label} · {image.size[0]}×{image.size[1]} → {result.size[0]}×{result.size[1]}',
            'width': result.size[0], 'height': result.size[1]}


if __name__ == '__main__':
    action, folder = sys.argv[1:]
    directory = Path(folder)
    data = json.loads((directory / 'request.json').read_text())
    if action == 'enhance':
        (directory / 'prompt.txt').write_text(enhance_prompt(data))
    else:
        started = time.monotonic()
        if action == 'download':
            result = download(directory, data)
        elif action == 'upscale':
            result = upscale_image(directory, data)
        else:
            result = generate_image(directory, data)
        result['seconds'] = round(time.monotonic() - started, 1)
        (directory / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        report(directory, '完成')
