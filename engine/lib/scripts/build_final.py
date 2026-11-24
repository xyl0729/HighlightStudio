"""End-to-end highlight window and subject-following crop track builder."""
import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scratch", "post"))
from build_anchor_rule import subject_anchor_v2
from build_ef import fit, feats
from fit_decode import logit_pred, ef_decode
from hlcut.geometry import place_box, safe_crop_width
from hlcut.io_utils import load_index, read_jsonl, resolve_video_path, video_meta
from hlcut.pipeline import Config, _anchor_track, label_end


def full_track(s, tw, th, pow_y, pow_x, headroom):
    n, fps, W, H, axis = s["n"], s["fps"], s["W"], s["H"], s["axis"]
    recs = []
    for r0 in s["recs"]:
        r = dict(r0)
        dets = np.array(r0["dets"]) if r0.get("dets") else np.zeros((0, 6))
        r["s_anchor"], r["s_weight"] = subject_anchor_v2(dets, axis, W, H, s["crop_len"], headroom, True,
                                                         0.5, 0.0, 0.0, pow_y if axis == "y" else pow_x)
        recs.append(r)
    track = _anchor_track(recs, n, fps, axis, W, H, Config(spatial="detect", smooth_sec=0.4))
    cw = s["crop_len"] if axis == "x" else safe_crop_width(W, H, tw, th)
    return [list(place_box(track[f], H / 2.0, W, H, cw, tw, th) if axis == "x"
                 else place_box(W / 2.0, track[f], W, H, cw, tw, th)) for f in range(n)]


def greedy_window(greedy, fps, n_use, sec):
    m = re.search(r"(\d+)\s*to\s*(\d+)", greedy or "")
    if not m:
        return None
    c = 0.5 * (int(m.group(1)) + int(m.group(2)))
    a = max(0, int(round((c - sec / 2) * fps)))
    b = min(n_use - 1, int(round((c + sec / 2) * fps)))
    return (a, b) if b > a else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--train-post", default="scratch/post/train_snap490.jsonl")
    ap.add_argument("--scale-y", type=float, default=1.0)
    ap.add_argument("--scale-x", type=float, default=1.7)
    ap.add_argument("--pow-y", type=float, default=12.0)
    ap.add_argument("--pow-x", type=float, default=2.0)
    ap.add_argument("--headroom", type=float, default=0.03)
    ap.add_argument("--long-sec", type=float, default=120.0, help="x-axis videos longer than this use the greedy window")
    ap.add_argument("--long-win", type=float, default=20.0)
    ap.add_argument("--model-params-b", type=float, default=None, help="write model_params_b / model_size_mb fields")
    ap.add_argument("--model-size-mb", type=float, default=None)
    ap.add_argument("--check", default=None, help="reference package to compare against")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w, mu, sd, ntr = fit(os.path.join(ROOT, args.train_post))
    print("EF logistic fitted on %d training clips" % ntr)
    dump = {str(r["video_id"]): r for r in read_jsonl(args.dump)}
    post = {str(r["video_id"]): r for r in read_jsonl(args.post)}
    cnt = {"ef": 0, "long": 0, "whole": 0, "nodump": 0, "noaxis": 0}
    out = []
    for it in load_index(args.index):
        vid = str(it["video_id"])
        tw, th = float(it["targetRatioWH"][0]), float(it["targetRatioWH"][1])
        s = dump.get(vid)
        if s is None:
            n, fps, W, H = video_meta(resolve_video_path(it, args.video_dir))
            axis = None
            cnt["nodump"] += 1
        else:
            n, fps, W, H, axis = s["n"], s["fps"], s["W"], s["H"], s["axis"]
        if axis in ("x", "y"):
            boxes = full_track(s, tw, th, args.pow_y, args.pow_x, args.headroom)
        else:
            cnt["noaxis"] += s is not None
            b0 = list(place_box(W / 2.0, H / 2.0, W, H, safe_crop_width(W, H, tw, th), tw, th))
            boxes = [b0] * n
        n_use = label_end(n, fps)
        p = post.get(vid)
        win = None
        if axis == "x" and n / fps > args.long_sec and p is not None:
            win = greedy_window(p.get("greedy"), fps, n_use, args.long_win)
            cnt["long"] += win is not None
        if win is None and p is not None and p.get("logp"):
            q = logit_pred(w, (feats(p, n, fps) - mu) / sd)
            win = ef_decode(q, n_use, max(1, int(fps / 4)), args.scale_x if axis == "x" else args.scale_y)
            cnt["ef"] += 1
        if win is None:
            win = (0, n_use - 1)
            cnt["whole"] += 1
        a, b = win
        rec = {"video_id": vid, "targetRatioWH": it["targetRatioWH"],
               "predictions": [{"frame": f, "bboxes": boxes[f]} for f in range(a, b + 1)]}
        if args.model_params_b is not None:
            rec["model_params_b"] = args.model_params_b
        if args.model_size_mb is not None:
            rec["model_size_mb"] = args.model_size_mb
        out.append(rec)
    print("windows: %s" % cnt)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %s (%d records)" % (args.out, len(out)))

    if args.check:
        ref = {str(r["video_id"]): r for r in read_jsonl(args.check)}
        same_win = same_all = 0
        for r in out:
            g = ref.get(r["video_id"])
            if g is None:
                continue
            fa = [p["frame"] for p in r["predictions"]]
            fb = [p["frame"] for p in g["predictions"]]
            same_win += fa == fb
            same_all += r["predictions"] == [{"frame": p["frame"], "bboxes": list(p["bboxes"])} for p in g["predictions"]]
        print("check vs %s: same frame set %d/%d, identical records %d/%d"
              % (args.check, same_win, len(out), same_all, len(out)))


if __name__ == "__main__":
    main()
