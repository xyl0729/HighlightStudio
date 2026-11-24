"""Apply the six-stage highlight and crop-track postprocessing pipeline."""
import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
signals = HERE / "signals/scratch"
lib = HERE / "lib"
sys.path[:0] = [str(HERE), str(lib), str(lib / "scripts"), str(lib / "scratch/post")]

from hlcut.spatial import smooth_track
from replay_contract import validate
from scratch.audiovisual_alignment_20261013_v1 import common
from scratch.spatial_track_reference.track_parity import own_track

SCALES = (2., 1.4, 1., .8, 1.7, 2.3, 2.6, 3.)
SIGMA_SEC = .5
CFG = dict(min_win_sec=8., min_gap_sec=2., min_seg_sec=1.5,
           valley_norm_max=.20, peak_norm_min=.55)

def edit_features(windows,q,fps):
    a,b=np.asarray(windows).T
    n=len(q)
    lengths=b-a+1
    prefix=np.r_[0.,q.cumsum()]
    masses=prefix[b+1]-prefix[a]
    dl=(lengths-lengths[0])/n
    dq=(masses-masses[0])/n
    u=np.log(n/fps)
    return np.column_stack([dl,dq,(a-a[0])/n,(b-b[0])/n,np.abs(dl),
                            dl*u,dq*u,dl*q.mean(),dq*q.mean(),dq*q.std()])

def select(scores):
    s=np.asarray(scores,float).copy()
    s[0]=0
    return int(np.argmax(s))

def slice_track(base,left,right):
    gaps = [(a+1,b) for a,b in zip(base,base[1:]) if b>a+1]
    if gaps:
        left = min(left,gaps[0][0]-1)
        right = max(right,gaps[-1][1])
    excluded = {i for a,b in gaps for i in range(a,b)}
    return [i for i in range(left,right+1) if i not in excluded]

def design(actions,q,fps):
    n = len(q)
    a,b = np.asarray([(fs[0],fs[-1]) for fs in actions]).T
    lengths = np.asarray([len(fs) for fs in actions])
    mass = np.array([q[fs].sum() for fs in actions])
    dl,dq = (lengths-lengths[0])/n,(mass-mass[0])/n
    u = np.log(n/fps)
    return np.column_stack([dl,dq,(a-a[0])/n,(b-b[0])/n,np.abs(dl),dl*u,dq*u,
                            dl*q.mean(),dq*q.mean(),dq*q.std()])

def scope(row, case):
    fs = [p['frame'] for p in row['predictions']]
    return (row['targetRatioWH']==[16,9] and case['W']>case['H']
            and all(p['bboxes']==[0,0,case['W']] for p in row['predictions'])
            and fs==list(range(fs[0],fs[-1]+1)))

def postprocess(boxes,W,fps):
    from scratch.spatial_stability_20261005_v1.core import filter_boxes
    from scratch.four_explore_20261011_v1.core import center_x
    b=filter_boxes(boxes,W,fps,'gaussian')
    b[:,0]=center_x(b[:,0].tolist(),b[:,2].tolist(),W)
    return b

def filter_crop(boxes, cuts, width, fps):
    out = boxes.copy()
    edges = [0,*cuts,len(boxes)]
    for a,b in zip(edges,edges[1:]):
        out[a:b,0] = np.clip(np.rint(smooth_track(boxes[a:b,0],SIGMA_SEC*fps)),0,width-boxes[a:b,2])
    return out

def features(base, detector, width, fps):
    x = np.asarray(base, float)[:, 0]
    y = np.asarray(detector, float)[:, 0]
    w = np.asarray(base, float)[:, 2]
    n = len(x)
    normalized = (x+w/2)/width
    gap = (y-x)/w
    signed = np.column_stack([.5-normalized, gap, (np.median(x)-x)/w,
                              (width-w-2*x)/w])
    position = np.column_stack([signed, signed*np.abs(gap[:, None])])
    k = max(1, round(.25*fps))
    i = np.arange(n)
    left, right = np.maximum(i-k, 0), np.minimum(i+k, n-1)
    seconds = np.maximum((right-left)/fps, 1/fps)
    velocity = (x[right]-x[left])/w/seconds
    detector_velocity = (y[right]-y[left])/w/seconds
    curve = (x[right]+x[left]-2*x)/w
    detector_curve = (y[right]+y[left]-2*y)/w
    motion = np.column_stack([velocity, detector_velocity, detector_velocity-velocity,
                              curve, detector_curve, velocity*np.abs(gap),
                              detector_velocity*np.abs(gap), curve*np.abs(velocity)])
    return np.column_stack([position, motion])

def apply(base, residual, width):
    result = np.asarray(base).copy()
    result[:, 0] = np.clip(np.rint(result[:, 0]+np.clip(residual, -.125, .125)*result[:, 2]),
                           0, width-result[:, 2])
    return result

def find_macro_valley(q, fps, min_win_sec=8.0, min_gap_sec=2.0, min_seg_sec=1.5, valley_norm_max=0.20, peak_norm_min=0.50):
    q_min, q_max = float(q.min()), float(q.max())
    if q_max - q_min < 0.12:
        return None
    norm_q = (q - q_min) / (q_max - q_min)
    k = max(1, int(fps * 0.5))
    smooth = np.convolve(norm_q, np.ones(k) / k, mode='same')

    is_low = smooth < valley_norm_max
    d = np.diff(np.r_[False, is_low, False].astype(np.int8))
    runs = [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]

    min_gap_frames = int(fps * min_gap_sec)
    min_seg_frames = int(fps * min_seg_sec)
    valid_runs = []
    for r_s, r_e in runs:
        gap_len = r_e - r_s
        if gap_len >= min_gap_frames:
            if r_s >= min_seg_frames and (len(smooth) - r_e) >= min_seg_frames:
                lp = float(smooth[:r_s].max())
                rp = float(smooth[r_e:].max())
                if lp >= peak_norm_min and rp >= peak_norm_min:
                    valid_runs.append((r_s, r_e, gap_len, lp, rp))

    if not valid_runs:
        return None
    valid_runs.sort(key=lambda x: x[2], reverse=True)
    return valid_runs[0]

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def rows(path):
    return {str(row["video_id"]): row for row in map(json.loads, path.read_bytes().splitlines())}

def model(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}

def run():
    meta = {str(row["video_id"]): row for row in read(HERE / "raw_manifest.json")["cases"]}
    base_rows = rows(WORK / "built/predictions.jsonl")
    posts = rows(signals / "temporal_tighten/post_test_merged.jsonl")
    detections = rows(signals / "fusai/detections.jsonl")
    assert len(meta) > 0 and set(meta) == set(base_rows) == set(posts) == set(detections)
    calibration = model(signals / "fusai/calibration.npz")
    adaptive = model(HERE / "learned/semifinal_adaptive.npz")
    selector = model(HERE / "learned/semifinal_selector.npz")
    motion = model(HERE / "learned/semifinal_motion.npz")
    native = common.P.arithmetic()
    result = copy.deepcopy(base_rows)
    counts = dict(scale=0, spatial=0, selector=0, adaptive=0, macro=0)
    for vid, row in result.items():
        case = meta[vid]
        n, fps, width = case["n"], case["fps"], case["W"]
        pred = row["predictions"]
        frames = [p["frame"] for p in pred]
        raw = native["features"](posts[vid], n, fps, use_pos=False)
        q0 = native["logit_pred"](calibration["w"], (raw-calibration["mu"])/calibration["sd"])
        filtered = np.maximum(q0-.05, 0)
        q = filtered if filtered.any() else q0
        step = max(1, int(fps/(8 if filtered.any() else 4)))
        end = native["label_end"](n, fps)
        def windows():
            return [tuple(map(int, native["ef_decode"](q, end, step, scale))) for scale in SCALES]
        def adaptive_window(candidates):
            unique = list(dict.fromkeys(candidates))
            return unique[select((edit_features(unique,q,fps)/adaptive["sd"]) @ adaptive["beta"])]
        def replace(fs, track):
            row["predictions"] = [{"frame": f, "bboxes": track[f].tolist()} for f in fs]

        spatial_path = signals / "spatial_object_ranker/final_v1/test_clips" / f"{vid}.json"
        if scope(row, case):
            lo, hi = adaptive_window(windows())
            row["predictions"] = [{"frame": f, "bboxes": [0,0,width]} for f in range(lo,hi+1)]
            counts["scale"] += 1
        elif row["targetRatioWH"] == [9,16] and spatial_path.exists():
            track = postprocess(np.array(read(spatial_path)["boxes"], int), width, fps)
            smooth = filter_crop(track, [], width, fps)
            track[:,0] = smooth[:,0]
            counts["spatial"] += 1
            candidates = windows()
            lengths = np.array([hi-lo+1 for lo,hi in candidates])
            ratios = lengths / lengths[SCALES.index(1.7)]
            a,b = frames[0],frames[-1]
            actions = [frames]
            for ratio in ratios:
                length = min(n,max(1,round((b-a+1)*ratio)))
                lo = min(n-length,max(0,round((a+b-length+1)/2)))
                actions.append(slice_track(frames,lo,lo+length-1))
            actions = [list(fs) for fs in dict.fromkeys(map(tuple, actions))]
            x = design(actions,q,fps)
            scores = (x/selector["rank_sd"]) @ selector["rank_beta"]
            pick = select(scores)
            if pick:
                other = np.delete(scores[1:],pick-1)
                margin = scores[pick]-max(0.,float(other.max())) if len(other) else scores[pick]
                gx = np.r_[x[pick],scores[pick],margin,len(scores)]
                gate = float((gx/selector["gate_sd"]) @ selector["gate_beta"] + selector["gate_intercept"])
                if gate >= float(selector["threshold"]):
                    frames = actions[pick]
                    counts["selector"] += 1
            detector = np.asarray(own_track(detections[vid],row["targetRatioWH"]), int)
            residual = (features(track,detector,width,fps)/motion["sd"]) @ motion["beta"]
            track = apply(track,residual,width)
            if frames == list(range(frames[0],frames[-1]+1)):
                lo,hi = adaptive_window(candidates)
                counts["adaptive"] += (lo,hi) != (frames[0],frames[-1])
                frames = list(range(lo,hi+1))
            replace(frames,track)

        pred = row["predictions"]
        frames = [p["frame"] for p in pred]
        if frames == list(range(frames[0],frames[-1]+1)) and len(frames)/fps >= CFG["min_win_sec"]:
            valley = find_macro_valley(filtered[frames[0]:frames[-1]+1], fps, **CFG)
            if valley:
                lo,hi = valley[:2]
                row["predictions"] = pred[:lo]+pred[hi:]
                counts["macro"] += 1

    validate(list(result.values()),meta)

    output = WORK / "final"
    output.mkdir(exist_ok=False)
    with (output / "predictions.jsonl").open("x", encoding="utf8", newline="\n") as stream:
        for row in result.values():
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps(dict(records=len(result), stages=counts)))

if __name__ == "__main__":
    run()
