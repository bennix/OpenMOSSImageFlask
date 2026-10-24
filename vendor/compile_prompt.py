"""Compile a caption or top-left Layout draft into a MOSS-Image prompt.

Coordinate conversion matches the official moss-image-prompt-enhancer:
draft boxes are absolute pixels from the top-left, and the model receives
centered yxyx boxes.
"""
import json
import math


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(value, allowed, required, location):
    _require(isinstance(value, dict), f'{location} must be an object')
    unknown = value.keys() - allowed
    missing = required - value.keys()
    _require(not unknown, f'{location}: unknown fields {sorted(unknown)}')
    _require(not missing, f'{location}: missing fields {sorted(missing)}')


def _text(value, location):
    _require(isinstance(value, str) and bool(value.strip()), f'{location} must be nonempty text')


def centered_box(box, width, height, location='bbox'):
    _require(isinstance(box, list) and len(box) == 4, f'{location} must be [x1,y1,x2,y2]')
    _require(all(type(value) in (int, float) and math.isfinite(value) for value in box),
             f'{location} must contain four finite numbers')
    x1, y1, x2, y2 = box
    _require(x1 < x2 and y1 < y2, f'{location} must have positive width and height')
    _require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
             f'{location} exceeds the {width}x{height} canvas')
    result = [math.floor(round(y1, 9)) - height // 2, math.floor(round(x1, 9)) - width // 2,
              math.ceil(round(y2, 9)) - height // 2, math.ceil(round(x2, 9)) - width // 2]
    _require(result[0] < result[2] and result[1] < result[3], f'{location} is too small at export precision')
    return result


def compile_prompt(draft):
    _require(isinstance(draft, dict), 'Draft must be a JSON object')
    if 'caption' in draft and 'items' not in draft:
        _fields(draft, {'caption'}, {'caption'}, 'Caption draft')
        _text(draft['caption'], 'caption')
        return draft['caption']
    _fields(draft, {'width', 'height', 'bbox_coord_space', 'bbox_format', 'summary', 'items'},
            {'width', 'height', 'bbox_coord_space', 'bbox_format', 'summary', 'items'}, 'Layout draft')
    width, height = draft['width'], draft['height']
    for name, value in (('width', width), ('height', height)):
        _require(type(value) is int and value > 0 and value % 16 == 0, f'{name} must be a positive multiple of 16')
    _require(draft['bbox_coord_space'] == 'abs', 'Draft bbox_coord_space must be abs')
    _require(draft['bbox_format'] == 'xyxy', 'Draft bbox_format must be xyxy')
    summary = draft['summary']
    _fields(summary, {'caption', 'meta', 'canvas', 'palette'}, {'caption'}, 'summary')
    _text(summary['caption'], 'summary.caption')
    items = draft['items']
    _require(isinstance(items, list) and bool(items), 'items must be a nonempty array')
    compiled = []
    for index, item in enumerate(items):
        _fields(item, {'label', 'caption', 'bbox', 'text'}, {'label', 'caption', 'bbox'}, f'items[{index}]')
        _require(item['label'] in ('object', 'character', 'text'), f'items[{index}].label is invalid')
        _text(item['caption'], f'items[{index}].caption')
        if item['label'] == 'text':
            _require('text' in item, f'items[{index}] requires text')
        if 'text' in item:
            _text(item['text'], f'items[{index}].text')
        compiled.append({key: centered_box(value, width, height, f'items[{index}].bbox') if key == 'bbox' else value
                         for key, value in item.items()})
    result = {'summary': summary, 'items': compiled, 'bbox_coord_space': 'abs_centralized', 'bbox_format': 'yxyx'}
    return json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def scale_box(region, width, height):
    """Map a box from the viewed image into the requested canvas."""
    x1 = region['x'] * width / region['source_width']
    y1 = region['y'] * height / region['source_height']
    x2 = (region['x'] + region['w']) * width / region['source_width']
    y2 = (region['y'] + region['h']) * height / region['source_height']
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, max(x1 + 1, x2)), min(height, max(y1 + 1, y2))
    return [x1, y1, x2, y2]


def region_item(kind, note, box):
    note = note.strip()
    if kind == 'text':
        return {'label': 'text', 'caption': 'Legible lettering that fits the box and matches the surrounding image. Render the text exactly.',
                'text': note, 'bbox': box}
    return {'label': 'object', 'caption': note, 'bbox': box}


def top_left_box(box, width, height):
    y1, x1, y2, x2 = box
    return [x1 + width // 2, y1 + height // 2, x2 + width // 2, y2 + height // 2]


def append_region(prompt, note, box, width, height, kind='edit'):
    """Add one region to a compiled layout. A plain caption starts a new layout."""
    try:
        layout = json.loads(prompt)
    except json.JSONDecodeError:
        layout = None
    if not isinstance(layout, dict) or layout.get('bbox_format') != 'yxyx' or not isinstance(layout.get('items'), list):
        return region_prompt(prompt, note, box, width, height, kind)
    items = []
    for item in layout['items']:
        restored = dict(item)
        restored['bbox'] = top_left_box(item['bbox'], width, height)
        items.append(restored)
    items.append(region_item(kind, note, box))
    summary = layout.get('summary') if isinstance(layout.get('summary'), dict) else {'caption': prompt}
    draft = {'width': width, 'height': height, 'bbox_coord_space': 'abs', 'bbox_format': 'xyxy',
             'summary': summary, 'items': items}
    return compile_prompt(draft)


def region_prompt(summary, note, box, width, height, kind='edit'):
    summary = summary.strip() or 'The requested image.'
    if kind == 'text':
        summary += ' Preserve the existing scene, lighting, and subjects. Add only this text inside the marked region.'
    elif kind == 'add':
        summary += ' Preserve the existing scene. Add only the described content inside the marked region.'
    else:
        summary += ' Preserve the existing scene outside the marked region. Revise only that region as described.'
    draft = {
        'width': width, 'height': height, 'bbox_coord_space': 'abs', 'bbox_format': 'xyxy',
        'summary': {'canvas': 'single', 'caption': summary},
        'items': [region_item(kind, note, box)],
    }
    return compile_prompt(draft)
