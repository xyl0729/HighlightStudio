"""Run the unchanged detector math for every indexed source, in isolated blocks."""
from cold_runtime import *
import subprocess

def run():
    cs=cases();rows=[];counts=[]
    for start in range(0,len(cs),32):
        end=min(len(cs),start+32);stage=f'detector_block{start:03d}'
        command=[sys.executable,str(CODE/'resume_detector_block.py'),'--start',str(start),'--end',str(end)]
        with (WORK/(stage+'.stdout.log')).open('xb') as out,(WORK/(stage+'.stderr.log')).open('xb') as err:
            rc=subprocess.call(command,cwd=CODE,stdout=out,stderr=err)
        write(WORK/(stage+'_EXIT.json'),dict(actual_exit=rc));assert rc==0,(stage,rc)
        proof=read(WORK/(stage+'_terminal.json'))['proof'];rows.extend(proof['rows']);counts.append(proof['parameters'])
    records=[read(x['path']) for x in rows]
    assert [x['video_id'] for x in records]==[x['video_id'] for x in cs]
    assert len(set(counts))==1
    dest=SIGNALS/'scratch/fusai/detections.jsonl';dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x',encoding='utf8') as f:
        for x in records:f.write(json.dumps(x,allow_nan=False)+'\n')
    write(WORK/'detector_seal.json',dict(records=len(rows),parameters=counts[0],SHA256=sha(dest)))
    return dict(records=len(rows),parameters=counts[0],sampled_frames=sum(len(x['recs']) for x in records))

if __name__=='__main__':raise SystemExit(finish('detector',run))
