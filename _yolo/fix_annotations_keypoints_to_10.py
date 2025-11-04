#!/usr/bin/env python3
import json,sys,os,shutil,collections

def fix_file(p):
    if not os.path.exists(p):
        print('file not found',p); return 1
    bak = p + '.annotations_fix.bak'
    shutil.copy(p,bak)
    print('backup created:',bak)
    d = json.load(open(p))
    ann = d.get('annotations',[])
    before = collections.Counter([len(a.get('keypoints',[]))//3 for a in ann])
    modified_cnt = 0
    modified_ids = []
    for a in ann:
        kp = a.get('keypoints',[])
        kcnt = len(kp)//3
        if kcnt==12:
            # remove last two triplets
            newkp = kp[:30]
        elif kcnt>12:
            # unexpected: truncate to first 10
            newkp = kp[:30]
        elif kcnt<10:
            # pad zeros to reach 10*3
            newkp = kp + [0]* (30 - len(kp))
        else:
            newkp = kp
        if newkp!=kp:
            a['keypoints'] = newkp
            # recompute num_keypoints as count of v>0
            num = 0
            for i in range(2, len(newkp), 3):
                try:
                    if float(newkp[i])>0:
                        num += 1
                except:
                    pass
            a['num_keypoints'] = num
            modified_cnt += 1
            modified_ids.append(a.get('id'))
    after = collections.Counter([len(a.get('keypoints',[]))//3 for a in ann])
    json.dump(d, open(p,'w'), ensure_ascii=False, indent=2)
    print('file updated:',p)
    print('before distribution:', dict(before))
    print('after distribution:', dict(after))
    print('modified annotations:', modified_cnt)
    if modified_cnt>0:
        print('sample modified ids:', modified_ids[:20])
    return 0

if __name__=='__main__':
    if len(sys.argv)<2:
        print('Usage: fix_annotations_keypoints_to_10.py <path_to_json>')
        sys.exit(2)
    sys.exit(fix_file(sys.argv[1]))
