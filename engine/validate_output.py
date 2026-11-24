"""Validate and seal generated output. No reference result is an inference input."""
from cold_runtime import *
import zipfile

def run():
    p=WORK/'final/predictions.jsonl';rows=list(map(json.loads,p.read_bytes().splitlines()))
    meta={c['video_id']:c for c in cases()};seen=set();frames=0
    for row in rows:
        v=str(row['video_id']);assert v in meta and v not in seen;seen.add(v)
        c=meta[v];rw,rh=row['targetRatioWH'];assert row['targetRatioWH']==c['targetRatioWH'];prior=-1
        for pred in row['predictions']:
            f=pred['frame'];x,y,w=pred['bboxes']
            assert type(f) is int and prior<f<c['n'];prior=f
            assert all(type(t) is int for t in (x,y,w)) and min(x,y)>=0 and w>0
            assert x+w<=c['W'] and y*rw+w*rh<=c['H']*rw;frames+=1
        assert prior>=0
    assert seen==set(meta)
    write(WORK/'PREDICTION_SEAL.json',dict(records=len(rows),frames=frames,SHA256=sha(p)))
    destination=WORK/'predictions.zip'
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as z:z.write(p,'predictions.jsonl')
    with zipfile.ZipFile(destination) as z:assert z.testzip() is None
    return dict(records=len(rows),frames=frames,output=str(destination),ZIP_SHA256=sha(destination))

if __name__=='__main__':raise SystemExit(finish('validate',run))
