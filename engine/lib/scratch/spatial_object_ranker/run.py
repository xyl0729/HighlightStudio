import os,tempfile;os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',os.path.join(tempfile.gettempdir(),'torchinductor_cache'));os.environ['OMP_NUM_THREADS']='2'
import json,time,numpy as np,torch
from pathlib import Path
from .core import proposals,features,offset_track
from .pipeline import nearest_index,choose_x
from .model import fit,predict
from hlcut.geometry import iou_xywh,derived_height
from scratch.unisal_crop.run import digest,check_frozen,validate_boxes
ROOT=Path(__file__).resolve().parents[2];D=Path(__file__).parent
LABELS=ROOT/'data/train/高光剪辑训练集/qvhighlights-videos/train.jsonl'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2)
def load_cases(m):
 originals={}
 for batch in ['run_v1','expansion_v1']:
  for c in read(ROOT/'scratch/unisal_crop'/batch/'inputs.json')['clips']:originals[c['video_id']]=c['baseline_boxes']
 loaded={}
 for c in m['cases']:
  v=c['video_id'];base=np.asarray(read(ROOT/c['unisal_path'])['boxes'],int);sample=read(ROOT/c['detector_path'])['sample'];recs=sample['recs'];ids=[r['fid'] for r in recs]
  assert ids==sorted(set(ids)) and len(base)==c['n'] and ids[0]==0 and ids[-1]<c['n']
  assert np.array_equal(base[:,1:],np.asarray(originals[v])[:,1:])
  loaded[v]=(base,np.asarray(originals[v],int),recs,ids)
 return loaded

def train_rows(cases,loaded):
 wanted={c['video_id'] for c in cases}
 # Validation targets are not retained or consulted during this fold training.
 labels={r['video_id']:r['cropRois'] for r in map(json.loads,LABELS.read_text(encoding='utf-8-sig').splitlines()) if r['video_id'] in wanted}
 X=[];Y=[];W=[];oracle=[];counts={}
 for c in cases:
  v=c['video_id'];base,yolo,recs,ids=loaded[v];gt=sorted(labels[v],key=lambda r:r[0]);loc=np.unique(np.linspace(0,len(gt)-1,min(48,len(gt))).round().astype(int));xx=[];yy=[]
  for gi in loc:
   f,truth=gt[int(gi)];det=recs[nearest_index(ids,f)].get('dets',[]);bx,by,bw=map(int,base[f]);yx=int(yolo[f,0]);pool=proposals(bx,yx,bw,c['W'],c['H'],det);ref=iou_xywh([bx,by,bw,derived_height(bw,9,16)],truth);gains=[]
   for x in pool:
    xx.append(features(x,bx,yx,bw,c['W'],c['H'],det));gain=iou_xywh([x,by,bw,derived_height(bw,9,16)],truth)-ref;yy.append(gain);gains.append(gain)
   oracle.append(max(gains))
  assert xx;X.extend(xx);Y.extend(yy);W.extend([1/len(xx)]*len(xx));counts[v]=len(xx)
 return np.asarray(X,np.float32),np.asarray(Y,np.float32),np.asarray(W,np.float32),{'candidate_rows':len(X),'clip_rows':counts,'training_only_mean_candidate_oracle_iou_gain':float(np.mean(oracle))}

def infer_case(c,data,model,mu,sd):
 base,yolo,recs,ids=data;pools=[];allx=[];spans=[]
 for f,rec in zip(ids,recs):
  bx,by,bw=map(int,base[f]);yx=int(yolo[f,0]);pool=proposals(bx,yx,bw,c['W'],c['H'],rec.get('dets',[]));pools.append(pool);start=len(allx)
  allx.extend(features(x,bx,yx,bw,c['W'],c['H'],rec.get('dets',[])) for x in pool);spans.append((start,len(allx)))
 scores=predict(model,mu,sd,np.asarray(allx,np.float32));selected=[choose_x(pool,scores[a:b]) for pool,(a,b) in zip(pools,spans)]
 out=offset_track(base,ids,selected,c['W'],c['fps']);assert np.array_equal(out[:,1:],base[:,1:]);validate_boxes(out.tolist(),dict(c,baseline_boxes=base.tolist()))
 return {'video_id':c['video_id'],'fold':c['fold'],'boxes':out.tolist(),'sample_ids':ids,'selected_x':selected,'changed_frames':int(np.count_nonzero(out[:,0]!=base[:,0]))}

def main():
 start=time.time();m=read(D/'inputs.json');check_frozen();assert digest(D/'PROTOCOL.md')==m['protocol_sha256'] and digest(D/'core.py')==m['core_sha256']
 for p,h in m['hashes'].items():assert digest(ROOT/p)==h
 for p,h in m['protected'].items():assert digest(ROOT/p)==h
 out=D/'oof_v1';out.mkdir(exist_ok=False);(out/'clips').mkdir();loaded=load_cases(m);predhash={}
 code={p.name:digest(p) for p in D.glob('*.py')};save(out/'run_manifest.json',{'code':code,'inputs_sha256':digest(D/'inputs.json'),'label_sha256':digest(LABELS),'protocol_sha256':digest(D/'PROTOCOL.md')})
 for fold in [0,1]:
  train=[c for c in m['cases'] if c['fold']!=fold];valid=[c for c in m['cases'] if c['fold']==fold];assert not ({c['group'] for c in train}&{c['group'] for c in valid})
  X,Y,W,info=train_rows(train,loaded);print('TRAIN',fold,len(train),len(X),flush=True);model,mu,sd,loss=fit(X,Y,W)
  torch.save({'state':model.state_dict(),'mu':mu.tolist(),'sd':sd.tolist(),'fold':fold},out/('fold'+str(fold)+'.pt'))
  save(out/('fold'+str(fold)+'_training.json'),dict(info,losses=loss,validation_records=len(valid),train_source_ids=sorted({c['group'] for c in train})))
  for k,c in enumerate(valid):
   row=infer_case(c,loaded[c['video_id']],model,mu,sd);p=out/'clips'/(c['video_id']+'.json');save(p,row);predhash[c['video_id']]=digest(p)
   if k%50==0:print('INFER',fold,k+1,'/',len(valid),flush=True)
  print('FOLD_COMPLETE',fold,flush=True)
 assert len(predhash)==621
 for p,h in m['protected'].items():assert digest(ROOT/p)==h
 save(out/'complete.json',{'records':621,'seconds':time.time()-start,'predictions_sha256':predhash,'inputs_sha256':digest(D/'inputs.json'),'labels_sha256':digest(LABELS)})
 print('OOF_COMPLETE',flush=True)
if __name__=='__main__':main()
