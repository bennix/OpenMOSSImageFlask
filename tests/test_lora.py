import json
from pathlib import Path
import pytest
from torch import nn
import torch
from safetensors.torch import save_file
import worker


def test_fuse_lora_scales_matching_linear(tmp_path):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.ModuleDict({
                'proj': nn.ModuleDict({'q': nn.Linear(4, 4, bias=False)}),
            })])

    net = Net()
    with torch.no_grad():
        net.layers[0].proj['q'].weight.copy_(torch.eye(4))
    matrix_a = torch.zeros(2, 4)
    matrix_a[0, 0] = 1
    matrix_b = torch.zeros(4, 2)
    matrix_b[0, 0] = 1
    save_file({
        'transformer.layers.0.proj.q.lora_A.weight': matrix_a,
        'transformer.layers.0.proj.q.lora_B.weight': matrix_b,
    }, tmp_path / 'pytorch_lora_weights.safetensors')
    (tmp_path / 'adapter_config.json').write_text(json.dumps({'r': 2, 'lora_alpha': 2}))
    assert worker.fuse_lora(net, tmp_path, 0.5) == 1
    assert abs(net.layers[0].proj['q'].weight[0, 0].item() - 1.5) < 1e-5
    assert abs(net.layers[0].proj['q'].weight[1, 1].item() - 1) < 1e-5


def test_downloaded_lora_covers_every_dit_projection():
    path = Path('models/lora/pytorch_lora_weights.safetensors')
    if not path.is_file():
        pytest.skip('LoRA 权重尚未下载')
    from safetensors import safe_open
    with safe_open(path, framework='pt') as handle:
        keys = set(handle.keys())
    names = ('q', 'k', 'v', 'o', 'ffn_up_gate', 'ffn_down')
    expected = {f'transformer.layers.{index}.proj.{name}.lora_{side}.weight'
                for index in range(42) for name in names for side in ('A', 'B')}
    assert keys == expected
