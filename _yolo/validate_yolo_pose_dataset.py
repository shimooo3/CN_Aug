#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', help='dataset directory that contains train/valid/test subfolders (or config.yaml parent)')
args = parser.parse_args()

base = Path(args.dataset_dir).expanduser().resolve()
if (base / 'config.yaml').exists():
    # if given dataset root (folder containing config.yaml), resolve path inside
    cfg = base / 'config.yaml'
    # try to read kpt_shape and train path from yaml minimally
    try:
        text = cfg.read_text()
        kpt_shape = None
        train_rel = 'train/images'
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('kpt_shape'):
                # expect like: kpt_shape: [12, 3]
                import re
                m = re.search(r"\[(\d+)\s*,\s*(\d+)\]", line)
                if m:
                    kpt_shape = (int(m.group(1)), int(m.group(2)))
            if line.startswith('train:'):
                train_rel = line.split(':',1)[1].strip()
        if kpt_shape is None:
            print('warning: kpt_shape not found in config.yaml; will infer from labels')
    except Exception as e:
        print('cannot read config.yaml', e)
else:
    # assume provided path is dataset root already
    cfg = None
    kpt_shape = None
    train_rel = 'train/images'

# resolve train images and labels directories
# dataset path resolution: if cfg exists, use config path: path: ... may refer to folder name
# If user passed the dataset folder already containing train, use that
if cfg:
    # read "path:" field
    cfg_text = cfg.read_text()
    dsname = None
    for line in cfg_text.splitlines():
        if line.strip().startswith('path:'):
            dsname = line.split(':',1)[1].strip()
            break
    if dsname:
        # if dsname is absolute or relative
        dsroot = (base.parent / dsname).resolve() if not Path(dsname).is_absolute() else Path(dsname).resolve()
    else:
        dsroot = base
else:
    dsroot = base

train_images = (dsroot / train_rel).resolve()
train_labels = (train_images.parent / 'labels').resolve()

print('Dataset root     :', dsroot)
print('Train images dir :', train_images)
print('Train labels dir :', train_labels)

if not train_images.exists():
    print('ERROR: train images dir not found')
    sys.exit(2)
if not train_labels.exists():
    print('ERROR: train labels dir not found')
    sys.exit(2)

img_files = sorted([p for p in train_images.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png']])
label_files = sorted([p for p in train_labels.iterdir() if p.suffix.lower()=='.txt'])

print('num images:', len(img_files))
print('num labels:', len(label_files))

# map basenames
img_bases = {p.stem:p for p in img_files}
label_bases = {p.stem:p for p in label_files}

missing_labels = [b for b in img_bases.keys() if b not in label_bases]
extra_labels = [b for b in label_bases.keys() if b not in img_bases.keys()]

if missing_labels:
    print('Images missing labels (sample up to 10):', missing_labels[:10])
else:
    print('All images have label files (by name).')
if extra_labels:
    print('Label files without matching images (sample up to 10):', extra_labels[:10])

# inspect label token counts
from collections import Counter
counts = Counter()
bad_lines = []
max_kpts = 0
for p in label_files:
    try:
        txt = p.read_text().strip()
    except Exception as e:
        bad_lines.append((p.name, 'read_error', str(e)))
        continue
    if not txt:
        counts['empty_file'] += 1
        continue
    for i, line in enumerate(txt.splitlines(), start=1):
        toks = line.split()
        counts['total_lines'] += 1
        counts['tokens_'+str(len(toks))] += 1
        if len(toks) < 5:
            bad_lines.append((p.name, i, 'too_few_tokens', len(toks)))
            continue
        kpts = len(toks) - 5
        if kpt_shape is not None:
            expected = kpt_shape[0]*kpt_shape[1]
            if kpts != expected:
                bad_lines.append((p.name, i, 'kpt_count_mismatch', kpts, expected))
        else:
            if kpts % 3 != 0:
                bad_lines.append((p.name, i, 'kpt_tokens_not_multiple_of_3', kpts))
            else:
                max_kpts = max(max_kpts, kpts//3)

# report
print('\nLabel tokens distribution sample:')
for k,v in counts.most_common():
    print(f'  {k}: {v}')

if bad_lines:
    print('\nFound label format issues (sample up to 20):')
    for item in bad_lines[:20]:
        print(' ', item)
else:
    print('\nNo obvious label format issues found in inspected files.')

if kpt_shape is None:
    print('\nInferred max keypoints per line (if any):', max_kpts)
    if max_kpts>0:
        print('Inferred kpt_shape would be [', max_kpts, ',3]')

print('\nDone')
