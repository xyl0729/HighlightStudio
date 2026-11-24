"""Reproduce the two delivered gap heads from archived official-TRAIN FIT matrices.
Does not regenerate the upstream pixel/label feature matrices and never trains on TEST.
"""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse,json,time
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
D=Path(__file__).resolve().parent
CONFIG=dict(max_iter=80,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10.,early_stopping=False,random_state=20261020)
def main():
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);args=a.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    reports=[];tick=time.monotonic()
    for i in (0,1):
        with np.load(D/f'training_evidence/outer{i}_main_FIT.npz',allow_pickle=False) as z:X=z['X'];margin=z['margin'];weight=z['weight'];fitgroups=set(z['groups'])
        with np.load(D/f'training_evidence/outer{i}_main_scores.npz',allow_pickle=False) as z:C=z['X'];calgroups=set(z['groups'])
        if fitgroups&calgroups:raise ValueError('SOURCE leakage')
        model=HistGradientBoostingClassifier(**CONFIG).fit(X,margin>0,sample_weight=weight)
        ref=joblib.load(D/f'learned/model{i}.joblib')
        e=float(np.max(np.abs(model.predict_proba(C)-ref.predict_proba(C))))
        if e!=0:raise ValueError('head probability not bit-exact: '+str(e))
        joblib.dump(model,out/f'model{i}.joblib',compress=3)
        reports.append(dict(member=i,FIT_rows=len(X),CAL_rows=len(C),iterations=model.n_iter_,max_probability_error=e,SOURCE_disjoint=True))
    r=dict(actual_exit=0,heads=reports,seconds=time.monotonic()-tick,upstream_feature_extraction_retrained=False,TEST_training=False)
    (out/'RESULT.json').write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r))
if __name__=='__main__':main()
