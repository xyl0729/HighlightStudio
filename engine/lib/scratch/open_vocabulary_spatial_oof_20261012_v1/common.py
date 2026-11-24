import sys,json,hashlib,importlib.metadata
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;R=D.parents[1];sys.path.insert(0,str(R))
ENGINEER=R/'scratch/open_vocabulary_detection_20261012_v1';DIAGNOSIS=R/'scratch/open_vocabulary_output_diagnosis_20261012_v1'
from scratch.open_vocabulary_detection_20261012_v1 import common as G
from scratch.spatial_path_rank_block_20261010_v3 import common as P
from scratch.spatial_path_rank_block_20261010_v3 import evaluate as E
from scratch.four_explore_20261011_v1.core import center_x
read=G.read;sha=G.sha;save=G.save;atomic=G.atomic
CONFIG=dict(prompt=G.PROMPT,threshold=.30,max_objects=6,nms_iou=.5,samples_per_clip=12,batch_cases=10,features=54,
    folds=2,epochs=30,seed=20261012,lr=.001,weight_decay=.001,fit_batch=4096,head_elements=5633,normalization_elements=108,
    bootstrap_seed=20261008,bootstrap_draws=20000,ci_quantiles=[.0125,.9875],joint_gate=.002,gpu_only=True)
def check(full_sources=False,extraction=False):
    m=read(D/'manifest.json')
    if m['config']!=CONFIG or m['versions']!=G.versions():raise ValueError('Frozen quality config/runtime drift')
    for rel,h in m['files'].items():
        if extraction and rel in m['unused_large_detection_asset_files']:continue
        if sha(R/rel)!=h:raise ValueError('Quality dependency changed:'+rel)
    if full_sources:
        for rel,h in m['source_hashes'].items():
            if sha(R/rel)!=h:raise ValueError('Original TRAIN bytes changed')
    return m
def baseline(c,fold,inputs):
    raw=np.array(read(R/inputs['folds'][str(fold)]['reference_dir']/(c['video_id']+'.json'))['boxes'],int)
    b=P.current_boxes(c,raw,c['current_window']);b[:,0]=center_x(b[:,0].tolist(),b[:,2].tolist(),c['W']);E._boxes(c,b.tolist());return b
def check_case(c):
    if not c['source'].startswith('data/train/') or sha(R/c['source'])!=c['source_sha256']:raise ValueError('Original source identity')
