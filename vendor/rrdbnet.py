"""RRDBNet and tiled upscaling for official Real-ESRGAN weights.

The layer names match BasicSR so the published .pth files load unchanged.
Inference stays in this process and does not import basicsr.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _blocks(block, count, **kwargs):
    return nn.Sequential(*[block(**kwargs) for _ in range(count)])


class _Dense(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = _Dense(num_feat, num_grow_ch)
        self.rdb2 = _Dense(num_feat, num_grow_ch)
        self.rdb3 = _Dense(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = _blocks(_RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, 4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_rrdb(path, blocks, scale, device):
    model = RRDBNet(3, 3, scale=scale, num_block=blocks)
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('params_ema') or checkpoint.get('params') or checkpoint
    model.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()}, strict=True)
    return model.eval().to(device)


@torch.no_grad()
def enhance(model, image, device, tile=400, tile_pad=10, progress=None):
    """Upscale a PIL image by the network scale. `tile` limits memory."""
    array = torch.from_numpy(np.array(image.convert('RGB'))).float().permute(2, 0, 1) / 255
    picture = array.unsqueeze(0).to(device)
    scale = model.scale
    if scale == 2:
        _, _, height, width = picture.shape
        pad_h = (2 - height % 2) % 2
        pad_w = (2 - width % 2) % 2
        if pad_h or pad_w:
            picture = F.pad(picture, (0, pad_w, 0, pad_h), mode='reflect')
    else:
        pad_h = pad_w = 0
    _, _, height, width = picture.shape
    output = picture.new_zeros(1, 3, height * scale, width * scale)
    tiles_x = max(1, math.ceil(width / tile))
    tiles_y = max(1, math.ceil(height / tile))
    total = tiles_x * tiles_y
    done = 0
    for y in range(tiles_y):
        for x in range(tiles_x):
            start_y, start_x = y * tile, x * tile
            end_y, end_x = min(start_y + tile, height), min(start_x + tile, width)
            pad_y0, pad_x0 = max(start_y - tile_pad, 0), max(start_x - tile_pad, 0)
            pad_y1, pad_x1 = min(end_y + tile_pad, height), min(end_x + tile_pad, width)
            region = picture[:, :, pad_y0:pad_y1, pad_x0:pad_x1]
            _, _, region_h, region_w = region.shape
            divisor = 2 if scale == 2 else 1
            extra_h = (divisor - region_h % divisor) % divisor
            extra_w = (divisor - region_w % divisor) % divisor
            if extra_h or extra_w:
                region = F.pad(region, (0, extra_w, 0, extra_h), mode='reflect')
            tile_out = model(region)
            if extra_h or extra_w:
                tile_out = tile_out[:, :, :region_h * scale, :region_w * scale]
            out_y0 = (start_y - pad_y0) * scale
            out_x0 = (start_x - pad_x0) * scale
            output[:, :, start_y * scale:end_y * scale, start_x * scale:end_x * scale] = tile_out[
                :, :, out_y0:out_y0 + (end_y - start_y) * scale, out_x0:out_x0 + (end_x - start_x) * scale]
            done += 1
            if progress:
                progress(done, total)
    if pad_h or pad_w:
        output = output[:, :, 0:output.shape[2] - pad_h * scale, 0:output.shape[3] - pad_w * scale]
    pixels = output.squeeze(0).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    from PIL import Image
    return Image.fromarray(pixels)
