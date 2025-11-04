#!/usr/bin/env python3
"""
Fix YOLOv8-pose label files to match expected kpt_shape.
Backs up label files to a directory before modifying.
Usage:
  python tools/fix_yolo_pose_labels.py /path/to/dataset_dir [--kpt 12 3]
If config.yaml exists in dataset_dir, it will try to read kpt_shape from it.
"""
from pathlib import Path
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir')
parser.add_argument('--kpt', nargs=2, type=int, metavar=('N','M'), help='kpt_shape N M, e.g. 12 3')
parser.add_argument('--dry', action='store_true', help='dry run, only report')
args = parser.parse_args()

base = Path(args.dataset_dir).expanduser().resolve()
cfg = base / 'config.yaml'
if cfg.exists():
    text = cfg.read_text()
    m = re.search(r"kpt_shape:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if m:
        kpt_shape = (int(m.group(1)), int(m.group(2)))
    else:
        kpt_shape = None
else:
    kpt_shape = None

if args.kpt:
    kpt_shape = (args.kpt[0], args.kpt[1])

if kpt_shape is None:
    print('kpt_shape not found. Please provide with --kpt N M or add to config.yaml')
    raise SystemExit(2)

expected_kpt_tokens = kpt_shape[0] * kpt_shape[1]
expected_tokens = 5 + expected_kpt_tokens

print('Using kpt_shape=', kpt_shape, 'expected_tokens_per_line=', expected_tokens)

train_images = (base / 'train' / 'images')
train_labels = (base / 'train' / 'labels')
if not train_labels.exists():
    print('train/labels not found under', base)
    raise SystemExit(2)

bak_dir = train_labels.parent / 'labels.bak'
if not bak_dir.exists():
    bak_dir.mkdir(parents=True)

modified = 0
for p in sorted(train_labels.glob('*.txt')):
    txt = p.read_text()
    if not txt.strip():
        continue
    lines = txt.splitlines()
    out_lines = []
    changed = False
    for line in lines:
        toks = line.split()
        if len(toks) < 5:
            # skip malformed header
            print('skip malformed line in', p.name, '->', line)
            out_lines.append(line)
            continue
        if len(toks) == expected_tokens:
            out_lines.append(' '.join(toks))
            continue
        if len(toks) < expected_tokens:
            # pad with zeros
            pad = ['0'] * (expected_tokens - len(toks))
            newt = toks + pad
            out_lines.append(' '.join(newt))
            changed = True
        else:
            # too many tokens: truncate to expected_tokens
            newt = toks[:expected_tokens]
            out_lines.append(' '.join(newt))
            changed = True
    if changed:
        print('modify', p.name)
        modified += 1
        if not args.dry:
            # backup original
            bak_path = bak_dir / p.name
            if not bak_path.exists():
                p.rename(bak_path)
            else:
                p.unlink()
                p.write_text('\n'.join(out_lines))
            # write new file at original path
            (train_labels / p.name).write_text('\n'.join(out_lines))

print('Done. files modified:', modified)
print('Backups under', bak_dir)
