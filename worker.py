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


def download(directory, data):
    from downloader import download_model
    spec = MODELS[data['model']]
    report(directory, '下载权重（支持断点续传）')
    download_model(spec, ROOT / 'models' / spec['folder'])
    return {'model': data['model']}


def enhance_prompt(prompt):
    from mlx_lm import load, generate
    model, tokenizer = load(str(ROOT / 'models' / 'prompt'))
    messages = [
        {'role': 'system', 'content': 'You write image captions for MOSS-Image. Expand the user idea into a coherent, detailed English image prompt. Preserve quoted text exactly in its original language. Describe subject, composition, lighting, colors, and style. Do not add sexual or explicit content. Output only the caption, without explanations.'},
        {'role': 'user', 'content': prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    result = generate(model, tokenizer, prompt=rendered, max_tokens=384, verbose=False)
    if not result.strip():
        raise RuntimeError('提示词模型返回空内容。')
    return result.strip()


def generate_image(directory, data):
    prompt = data['prompt']
    if data['enhance']:
        # A separate process guarantees all MLX allocations are released before MOSS loads.
        report(directory, 'MLX 提示词扩写')
        import subprocess
        subprocess.run([sys.executable, __file__, 'enhance', str(directory)], check=True)
        prompt = (directory / 'prompt.txt').read_text()
    else:
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
    negative, negative_mask = pipe.encode_prompt('', max_sequence_length=512)
    pipe.text_encoder = None
    pipe.tokenizer = None
    del encoder, tokenizer
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    elif device == 'cuda':
        torch.cuda.empty_cache()

    report(directory, '加载图像模型')
    pipe.transformer = MOSSImageTransformer.from_pretrained(model_path / 'transformer',
                        torch_dtype=dtype, local_files_only=True).eval().to(device)
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
                 guidance_scale=3.0, uncond_skip_layers=[22],
                 generator=torch.Generator('cpu').manual_seed(data['seed'])).images[0]
    image.save(directory / 'image.png')
    return {'image': f'/outputs/{directory.name}/image.png', 'prompt': prompt, 'device': device,
            'width': data['width'], 'height': data['height'], 'steps': data['steps'], 'seed': data['seed']}


if __name__ == '__main__':
    action, folder = sys.argv[1:]
    directory = Path(folder)
    data = json.loads((directory / 'request.json').read_text())
    if action == 'enhance':
        (directory / 'prompt.txt').write_text(enhance_prompt(data['prompt']))
    else:
        started = time.monotonic()
        result = download(directory, data) if action == 'download' else generate_image(directory, data)
        result['seconds'] = round(time.monotonic() - started, 1)
        (directory / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        report(directory, '完成')
