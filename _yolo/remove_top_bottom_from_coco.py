#!/usr/bin/env python3
import json,collections,shutil,sys
p = '/home/shimoguchi/spring/AI_aug_gen/__dataset__/05_generate_dataset/00_temp/1103_daytime_unetDecoder01_rand/1103_daytime_unetDecoder01_rand.json'
# load
with open(p,'r') as f:
    d = json.load(f)
# backup
bak = p + '.bak'
shutil.copy(p,bak)
print('backup created:', bak)
modified = False
for cat in d.get('categories',[]):
    if 'keypoints' in cat:
        kp = cat['keypoints']
        # find positions (1-based) of 'top' and 'bottom'
        positions = {name: i+1 for i,name in enumerate(kp)}
        to_remove_names = [n for n in ('top','bottom') if n in positions]
        if not to_remove_names:
            continue
        to_remove_positions = set(positions[n] for n in to_remove_names)
        new_kp = [n for n in kp if n not in to_remove_names]
        new_skel = [pair for pair in cat.get('skeleton',[]) if pair[0] not in to_remove_positions and pair[1] not in to_remove_positions]
        print('category id', cat.get('id'), 'removed', to_remove_names, 'old_len', len(kp), 'new_len', len(new_kp))
        cat['keypoints'] = new_kp
        cat['skeleton'] = new_skel
        modified = True
# write back
if modified:
    with open(p,'w') as f:
        json.dump(d,f,ensure_ascii=False,indent=2)
    print('file updated:', p)
else:
    print('no categories changed')
# validation summary
cnt = collections.Counter([len(a.get('keypoints',[]))//3 for a in d.get('annotations',[])])
print('annotation keypoint counts distribution:', dict(cnt))
for i,cat in enumerate(d.get('categories',[])):
    print('cat index',i,'id',cat.get('id'),'keypoints_len',len(cat.get('keypoints',[])))
# quick check: ensure most annotations len matches categories kpts length (if single category assumed)
if len(d.get('categories',[]))>=1:
    want = len(d['categories'][0].get('keypoints',[]))
    total = len(d.get('annotations',[]))
    ok = sum(1 for a in d.get('annotations',[]) if len(a.get('keypoints',[]))//3==want)
    print('annotations matching new category keypoint length:', ok, '/', total)

print('done')
