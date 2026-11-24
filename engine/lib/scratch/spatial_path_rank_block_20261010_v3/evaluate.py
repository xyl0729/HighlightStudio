"""Complete sealed three-arm pilot evaluation on historical model annotations."""
import json
from pathlib import Path
import numpy as np
try:
    from . import common as C
except ImportError:
    import common as C

ARMS = ('baseline', 'control', 'candidate')


def checked_population(inputs):
    cases = inputs['cases']
    if len(cases) != 621 or len({c['video_id'] for c in cases}) != 621 or len({c['group'] for c in cases}) != 564:
        raise ValueError('Evaluation requires the complete original 621 clips / 564 sources')
    groups, sources = {}, {}
    for c in cases:
        vid, group, fold = c['video_id'], c['group'], c['fold']
        if not isinstance(vid,str) or not vid or Path(vid).name != vid or vid in ('.','..') or not isinstance(group,str) or not group:
            raise ValueError('Invalid frozen case identity')
        if type(fold) is not int or fold not in (0,1) or groups.setdefault(group,fold) != fold:
            raise ValueError('Source group crosses folds')
        digest = c['source_sha256']
        if not isinstance(digest,str) or len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest) or sources.setdefault(digest,group)!=group:
            raise ValueError('Frozen source hash missing or source alias crosses groups')
        if any(type(c[k]) is not int or c[k]<=0 for k in ('n','W','H')) or not np.isfinite(c['fps']) or c['fps']<=0:
            raise ValueError('Invalid native frame clock/dimensions')
        _window(c['current_window'],c['n'])
    if set(inputs['folds']) != {'0','1'}:
        raise ValueError('Both frozen folds required')
    for f in (0,1):
        fold = inputs['folds'][str(f)]
        for prefix,selected in (('train',[c for c in cases if c['fold']!=f]),('valid',[c for c in cases if c['fold']==f])):
            ids, gs = fold[prefix+'_ids'], fold[prefix+'_groups']
            if not selected or len(ids)!=len(set(ids)) or set(ids)!={c['video_id'] for c in selected} or len(gs)!=len(set(gs)) or set(gs)!={c['group'] for c in selected}:
                raise ValueError('Frozen fold population differs')
    return cases


def _window(window,n):
    if not isinstance(window,list) or len(window)!=2 or any(type(x) is not int for x in window) or not 0<=window[0]<=window[1]<n:
        raise ValueError('Invalid native selected window')
    return window


def _boxes(c,boxes):
    if not isinstance(boxes,list) or len(boxes)!=c['n'] or any(not isinstance(b,list) or len(b)!=3 or any(type(v) is not int for v in b) for b in boxes):
        raise ValueError('Full native integer N x 3 boxes required')
    a=np.asarray(boxes,dtype=np.float64)
    if not np.isfinite(a).all() or np.any(a[:,:2]<0) or np.any(a[:,2]<=0) or np.any(a[:,0]+a[:,2]>c['W']) or np.any(a[:,1]*9+a[:,2]*16>c['H']*9):
        raise ValueError('Strict continuous submission geometry illegal')
    return a


def checked_record(c,record):
    if any(record.get(k)!=c[k] for k in ('video_id','group','fold')) or record.get('window')!=c['current_window']:
        raise ValueError('Prediction native identity/time differs')
    _window(record['window'],c['n'])
    tracks={arm:_boxes(c,record[arm+'_boxes']) for arm in ARMS}
    for arm in ARMS:
        if arm+'_window' in record and record[arm+'_window'] != c['current_window']:
            raise ValueError('Arm changed selected times')
        if not np.array_equal(tracks[arm][:,1:],tracks['baseline'][:,1:]):
            raise ValueError('Arm changed reference y/width')
    return tracks


def load_sealed(inputs,complete,directory,manifest_sha256):
    """Verify every complete record and final paired model before opening labels."""
    cases=checked_population(inputs); directory=Path(directory)
    wanted={c['video_id'] for c in cases}
    if complete.get('records')!=621 or complete.get('manifest_sha256')!=manifest_sha256 or set(complete.get('record_hashes',{}))!=wanted:
        raise ValueError('Complete OOF seal/population/manifest mismatch')
    keys={f'fold{f}_{arm}' for f in (0,1) for arm in ('candidate','control')}
    if set(complete.get('model_hashes',{})) != keys:
        raise ValueError('Complete paired model identities missing')
    model_digests={}
    for f in (0,1):
        actual=C.sha(directory/f'fold{f}'/'final.pt'); model_digests[f]=actual
        if any(complete['model_hashes'][f'fold{f}_{arm}']!=actual for arm in ('candidate','control')):
            raise ValueError('Final paired checkpoint hash differs')
    if {p.stem for p in (directory/'predictions').glob('*.json')}!=wanted:
        raise ValueError('Predictions are a subset or contain extra clips')
    records={}
    for c in cases:
        vid=c['video_id']; path=directory/'predictions'/(vid+'.json')
        if C.sha(path)!=complete['record_hashes'][vid]:
            raise ValueError('Sealed prediction hash differs: '+vid)
        r=C.read(path)
        if r.get('manifest_sha256',manifest_sha256)!=manifest_sha256 or r.get('model_sha256',model_digests[c['fold']])!=model_digests[c['fold']]:
            raise ValueError('Per-record manifest/model identity differs')
        if 'sequence_sha256' in r:
            sequence=directory/f"fold{c['fold']}"/'infer'/(vid+'.npz')
            if not sequence.is_file() or C.sha(sequence)!=r['sequence_sha256']:
                raise ValueError('Per-record immutable inference sequence identity differs')
        checked_record(c,r); records[vid]=r
    return records


def checked_truth(c,label):
    if label.get('video_id')!=c['video_id'] or label.get('targetRatioWH') != [9,16]:
        raise ValueError('Historical truth identity/ratio differs')
    source=label.get('provenance',{}).get('source_group')
    if not isinstance(source,str) or source.rsplit('_',2)[0]!=c['group'] or Path(c['source']).stem!=source:
        raise ValueError('Historical truth source provenance differs')
    truth={}
    rois=label.get('cropRois')
    if not isinstance(rois,list) or not rois:
        raise ValueError('Missing spatial truth; cannot create negatives')
    for item in rois:
        if not isinstance(item,list) or len(item)!=2 or type(item[0]) is not int or not 0<=item[0]<c['n'] or item[0] in truth:
            raise ValueError('Duplicate/invalid native truth frame')
        box=item[1]
        if not isinstance(box,list) or len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in box):
            raise ValueError('Malformed historical truth geometry')
        b=np.asarray(box,dtype=np.float64)
        if not np.isfinite(b).all() or np.any(b[:2]<0) or np.any(b[2:]<=0) or b[0]+b[2]>c['W'] or b[1]+b[3]>c['H']:
            raise ValueError('Historical truth outside native source')
        truth[item[0]]=b
    segments=label.get('segments'); covered=set()
    if not isinstance(segments,list) or not segments:
        raise ValueError('Missing historical truth segments')
    for seg in segments:
        if not isinstance(seg,dict): raise ValueError('Malformed historical segment')
        a,b=seg.get('start_frame'),seg.get('end_frame')
        if type(a) is not int or type(b) is not int or not 0<=a<=b<c['n']:
            raise ValueError('Historical segment outside native clock')
        covered.update(range(a,b+1))
    if covered!=set(truth):
        raise ValueError('Historical segment / crop ROI coverage differs')
    return truth


def _iou(a,b):
    intersection=max(0.,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0.,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    return float(intersection/(a[2]*a[3]+b[2]*b[3]-intersection))


def metric(c,boxes,truth):
    a,b=_window(c['current_window'],c['n']); track=_boxes(c,boxes)
    overlap=[f for f in truth if a<=f<=b]
    total=sum(_iou([*track[f],track[f,2]*16/9],truth[f]) for f in overlap)
    selected=b-a+1; denominator=selected+len(truth)
    return {'joint':float(2*total/denominator),'temporal':float(2*len(overlap)/denominator),
            'conditional_iou':float(total/len(overlap)) if overlap else 0.,
            'gt_frames':len(truth),'predicted_frames':selected,'overlap_frames':len(overlap)}


def score_record(c,record,label):
    checked_record(c,record); truth=checked_truth(c,label)
    row={k:c[k] for k in ('video_id','group','fold')}
    row['window']=list(c['current_window'])
    for arm in ARMS: row[arm]=metric(c,record[arm+'_boxes'],truth)
    row['delta_baseline']=row['candidate']['joint']-row['baseline']['joint']
    row['delta_control']=row['candidate']['joint']-row['control']['joint']
    row['delta_temporal']=row['candidate']['temporal']-row['baseline']['temporal']
    return row


def source_ci(delta,groups):
    values=np.asarray(delta,dtype=np.float64); unique=sorted(set(groups))
    if values.shape!=(len(groups),) or not unique or not np.isfinite(values).all(): raise ValueError('Invalid source bootstrap rows')
    group_array=np.asarray(groups)
    sums=np.array([values[group_array==g].sum() for g in unique]); counts=np.array([np.count_nonzero(group_array==g) for g in unique])
    rng=np.random.RandomState(C.CONFIG['bootstrap_seed']); draws=np.empty(C.CONFIG['bootstrap_draws'])
    # Retain the old source-resample arithmetic: a group contributes all of its clips.
    for i in range(len(draws)):
        chosen=rng.randint(len(unique),size=len(unique)); draws[i]=sums[chosen].sum()/counts[chosen].sum()
    return np.quantile(draws,C.CONFIG['ci_quantiles']).tolist()


def summarize(rows):
    if len(rows)!=621 or len({r['video_id'] for r in rows})!=621 or len({r['group'] for r in rows})!=564 or {r['fold'] for r in rows}!={0,1}:
        raise ValueError('Summary cannot score a selected subset')
    for r in rows:
        if any(r[arm]['temporal'] != r['baseline']['temporal'] for arm in ARMS) or r['delta_temporal']!=0.:
            raise ValueError('Spatial pilot changed temporal score')
        for name,other in (('delta_baseline','baseline'),('delta_control','control')):
            if r[name]!=r['candidate']['joint']-r[other]['joint']:
                raise ValueError('Row delta arithmetic differs')
        if any(not np.isfinite(r[arm][m]) or not 0<=r[arm][m]<=1 for arm in ARMS for m in ('joint','temporal','conditional_iou')):
            raise ValueError('Nonfinite/out-of-range row metric')
    groups=[r['group'] for r in rows]
    means={arm:{m:float(np.mean([r[arm][m] for r in rows])) for m in ('joint','temporal','conditional_iou')} for arm in ARMS}
    contrasts={other:{'delta':float(np.mean([r['delta_'+other] for r in rows])),
                      'ci97_5_by_source':source_ci([r['delta_'+other] for r in rows],groups)} for other in ('baseline','control')}
    folds={str(f):{'records':sum(r['fold']==f for r in rows),
                  'source_groups':len({r['group'] for r in rows if r['fold']==f}),
                  'delta_baseline':float(np.mean([r['delta_baseline'] for r in rows if r['fold']==f])),
                  'delta_control':float(np.mean([r['delta_control'] for r in rows if r['fold']==f]))} for f in (0,1)}
    base,control=contrasts['baseline'],contrasts['control']
    gates={'candidate_delta_at_least_001':base['delta']>=C.CONFIG['joint_gate'],
           'candidate_baseline_lower_ci_positive':base['ci97_5_by_source'][0]>0,
           'each_fold_candidate_baseline_positive':all(folds[str(f)]['delta_baseline']>0 for f in (0,1)),
           'candidate_control_lower_ci_positive':control['ci97_5_by_source'][0]>0,
           'temporal_delta_exact_zero':True,'complete_coverage_and_geometry':True}
    return {'records':621,'source_groups':564,'means':means,'candidate_vs_baseline':base,'candidate_vs_control':control,
            'folds':folds,'bootstrap':{k:C.CONFIG[k] for k in ('bootstrap_seed','bootstrap_draws','ci_quantiles')},
            'temporal_unchanged':True,'width_y_unchanged':True,'gates':gates,'pilot_gate_passed':all(gates.values()),
            'new_package_recommended':False,'actual_platform_score':None,
            'evidence':'Repeated historical model-generated training labels; development pilot only.'}


def main():
    manifest_sha=C.check(sequence_files=True); inputs=C.read(C.D/'inputs.json')
    complete_path=C.D/'oof_complete.json'; complete_sha=C.sha(complete_path); complete=C.read(complete_path)
    records=load_sealed(inputs,complete,C.D,manifest_sha)
    labels=C.selected_labels(set(records))
    rows=[score_record(c,records[c['video_id']],labels[c['video_id']]) for c in inputs['cases']]
    # Guard against changes during evaluation before publishing any score.
    if C.check(sequence_files=True)!=manifest_sha or C.sha(complete_path)!=complete_sha:
        raise ValueError('Frozen evaluation identities changed')
    load_sealed(inputs,complete,C.D,manifest_sha)
    result={'summary':summarize(rows),'rows':rows,'manifest_sha256':manifest_sha,'complete_sha256':complete_sha,
            'new_package_recommended':False,'actual_platform_score':None}
    C.atomic(C.D/'evaluation.json',result)
    print(json.dumps(result['summary'],ensure_ascii=False),flush=True)
    return result


if __name__=='__main__': main()
