"""Isolated, immutable source/sequence identities and native baseline arithmetic."""
import ast, hashlib, importlib.metadata, json, math
from pathlib import Path
import sys
import numpy as np

D = Path(__file__).resolve().parent
R = D.parents[1]
if str(R) not in sys.path: sys.path.insert(0, str(R))
LABEL = R / 'data/train/高光剪辑训练集/qvhighlights-videos/train.jsonl'
POST = R / 'scratch/post/train_all986.jsonl'
BEST = R / 'outputs/fusai/submission_fusai_fl05_step8_xy_20261006_v1.zip'
BEST_SHA = 'b3a4937bbc7814dc649e4d046ea6231681e641e5f0a5d3972046ba0d256eb923'
CONFIG = {'solver': 'fixed_baseline_continuity_aggregate_master', 'dtype': 'float64', 'l2': .001,
          'max_outer': 100, 'gap': .001, 'inner_maxiter': 1000, 'inner_ftol': 1e-12,
          'seed': 20261010, 'bootstrap_seed': 20261008, 'bootstrap_draws': 20000,
          'ci_quantiles': [.0125, .9875], 'joint_gate': .01, 'threads': 2}

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def atomic(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8'); tmp.replace(path)
def save(path, value):
    with Path(path).open('x', encoding='utf8') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
def versions(): return {p: importlib.metadata.version(p) for p in ('torch','numpy','scipy','opencv-python','transformers','bitsandbytes')}
def check(sequence_files=False):
    m = read(D/'manifest.json')
    if sha(BEST) != BEST_SHA: raise ValueError('protected scored package changed')
    if m['config'] != CONFIG or m['versions'] != versions(): raise ValueError('frozen config/runtime differs')
    for rel, digest in m['files'].items():
        if sha(R/rel) != digest: raise ValueError('frozen file changed:'+rel)
    if sequence_files:
        for rel, digest in read(D/'prepared_complete.json')['sequence_hashes'].items():
            if sha(D/rel) != digest: raise ValueError('prepared sequence changed:'+rel)
    return sha(D/'manifest.json')

def save_sequence(path, seq):
    sizes = [len(x) for x in seq['x']]; offsets = np.r_[0,np.cumsum(sizes)].astype(np.int32)
    meta = {k:v for k,v in seq.items() if k not in ('x','X','appearance','target_iou')}
    args = dict(metadata=np.array(json.dumps(meta,allow_nan=False)), offsets=offsets,
                x=np.concatenate(seq['x']).astype(np.int32), X=np.concatenate(seq['X']).astype(np.float32),
                appearance=np.concatenate(seq['appearance']).astype(np.float32))
    if 'target_iou' in seq: args['target_iou'] = np.concatenate(seq['target_iou']).astype(np.float32)
    with Path(path).open('xb') as f: np.savez_compressed(f, **args)
def load_sequence(path):
    with np.load(path, allow_pickle=False) as a:
        seq=json.loads(str(a['metadata'])); bounds=list(zip(a['offsets'][:-1],a['offsets'][1:]))
        for key in ('x','X','appearance','target_iou'):
            if key in a:
                flat=a[key]
                seq[key]=[flat[lo:hi].copy() for lo,hi in bounds]
    return seq
def selected_labels(wanted):
    import re
    out={}
    with LABEL.open(encoding='utf-8-sig') as f:
        for line in f:
            match=re.search(r'"video_id"\s*:\s*("(?:[^"\\]|\\.)*")',line)
            if match is None: raise ValueError('missing annotation identity')
            vid=json.loads(match.group(1))
            if vid in wanted:
                if vid in out: raise ValueError('duplicate annotation identity')
                out[vid]=json.loads(line)
    if set(out)!=set(wanted): raise ValueError('missing selected annotation')
    return out
def arithmetic():
    path=R/'scratch/timelens2_cloud/validate_cloud.py'
    names={'post_probs','_widen','features','logit_pred','ef_decode','label_end'}
    functions=[n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != names: raise ValueError('native arithmetic incomplete')
    ns={'np':np,'math':math,'WIDTHS':(.3,.6)}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),ns); return ns
def baseline_window(case, post, old, native):
    if post['n']!=case['n'] or abs(post['fps']-case['fps'])>1e-6: raise ValueError('native posterior clock differs')
    q=native['logit_pred'](old['w'],(native['features'](post,case['n'],case['fps'],use_pos=False)-old['mu'])/old['sd'])
    q=np.maximum(q-.05,0)
    if not np.isfinite(q).all(): raise ValueError('nonfinite native probabilities')
    # The deployed step8 builder keeps its fl05 window if floor leaves zero mass.
    # Reconstruct that fl05 fallback on the original fps/4 grid, never inputs.window.
    divisor=4 if q.sum()<=0 else 8
    a,b=native['ef_decode'](q,native['label_end'](case['n'],case['fps']),max(1,int(case['fps']/divisor)),1.7)
    if not 0<=a<=b<case['n']: raise ValueError('illegal native EF window')
    return [int(a),int(b)]
def current_boxes(case, raw_boxes, window):
    from scratch.spatial_stability_20261005_v1.core import filter_boxes
    from scratch.selected_y_smooth_20261005_v1.core import smooth_y_row
    raw=np.asarray(raw_boxes,int)
    if raw.shape!=(case['n'],3): raise ValueError('incomplete source-safe track')
    smooth=filter_boxes(raw,case['W'],case['fps'],'gaussian')
    a,b=window
    row={'video_id':case['video_id'],'targetRatioWH':[9,16],'model_params_b':5.19,
         'predictions':[{'frame':i,'bboxes':raw[i].tolist()} for i in range(a,b+1)]}
    filtered=smooth_y_row(row,case)
    for item in filtered['predictions']: smooth[item['frame'],1]=item['bboxes'][1]
    if not np.array_equal(smooth[:,2],raw[:,2]): raise ValueError('width changed')
    if np.any(smooth<0) or np.any(smooth[:,0]+smooth[:,2]>case['W']) or np.any(smooth[:,1]*9+smooth[:,2]*16>case['H']*9): raise ValueError('continuous geometry illegal')
    return smooth
