import json
from pathlib import Path
import pytest
from PIL import Image
from vendor.compile_prompt import compile_prompt
from vendor.rrdbnet import RRDBNet, enhance, load_rrdb


def test_layout_centers_the_official_cup_box():
    prompt = compile_prompt({
        'width': 1536, 'height': 1024, 'bbox_coord_space': 'abs', 'bbox_format': 'xyxy',
        'summary': {'caption': 'A warm studio campaign.'},
        'items': [{'label': 'object', 'caption': 'An ivory ceramic cup.', 'bbox': [176, 224, 800, 864]}],
    })
    assert json.loads(prompt)['items'][0]['bbox'] == [-288, -592, 352, 32]


def test_region_box_scales_onto_the_canvas():
    from vendor.compile_prompt import region_prompt, scale_box
    box = scale_box({'x': 128, 'y': 64, 'w': 256, 'h': 128, 'source_width': 512, 'source_height': 512}, 1024, 1024)
    assert box == [256.0, 128.0, 768.0, 384.0]
    compiled = json.loads(region_prompt('a forest cat', 'mossy stone', box, 1024, 1024))
    assert compiled['bbox_coord_space'] == 'abs_centralized'
    assert compiled['items'][0]['bbox'] == [-384, -256, -128, 256]
    lettered = json.loads(region_prompt('a forest cat', '青苔', box, 1024, 1024, 'text'))
    assert lettered['items'][0]['label'] == 'text' and lettered['items'][0]['text'] == '青苔'


def test_later_edit_keeps_the_previous_region():
    from vendor.compile_prompt import append_region, region_prompt
    first = region_prompt('a forest cat', 'sunglasses', [100, 80, 300, 200], 512, 512, 'add')
    second = json.loads(append_region(first, '青苔', [40, 40, 120, 90], 512, 512, 'text'))
    assert second['items'][0]['bbox'] == json.loads(first)['items'][0]['bbox']
    assert second['items'][0]['caption'] == 'sunglasses'
    assert second['items'][1]['text'] == '青苔' and second['items'][1]['label'] == 'text'


def test_tiled_x2_keeps_output_size():
    model = RRDBNet(3, 3, scale=2, num_block=1).eval()
    image = Image.new('RGB', (15, 17), (1, 2, 3))
    assert enhance(model, image, 'cpu', tile=8, tile_pad=1).size == (30, 34)


def test_x2_weights_load():
    path = Path('models/upscale/RealESRGAN_x2plus.pth')
    if not path.is_file():
        pytest.skip('放大权重尚未下载')
    model = load_rrdb(path, 23, 2, 'cpu')
    assert model.conv_first.in_channels == 12


def test_official_anime_weights_upscale_a_tile():
    path = Path('models/upscale/RealESRGAN_x4plus_anime_6B.pth')
    if not path.is_file():
        pytest.skip('放大权重尚未下载')
    model = load_rrdb(path, 6, 4, 'cpu')
    result = enhance(model, Image.new('RGB', (16, 16), (20, 40, 60)), 'cpu', tile=8, tile_pad=2)
    assert result.size == (64, 64)
