"""Expected-F window decode from the answer posterior with multiple feature sources.
A logistic model q(frame in GT | features) is fitted on training clips, then each test video's
window [a, b] is decoded to maximise expected F on that video. The boxes are pulled from
full-length trajectories (full169 / full916).

  python scratch/post/build_ef.py \
      --base outputs/submission_vlm_mbr169.jsonl \
      --full169 scratch/full169_p12.jsonl \
      --full916 outputs/submission_vlm_916_longcut.jsonl \
      --scale 1.0 --ratio 169 --out outputs/submission_vlm_ef169.jsonl
"""
import argparse, glob, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_big import load_gt, load_post, gt_mask
from fit_decode import features, logit_fit, logit_pred, ef_decode
from hlcut.io_utils import video_meta
from hlcut.pipeline import label_end


def load(p):
    return {str(json.loads(l)["video_id"]): json.loads(l) for l in open(p, encoding="utf-8")}


PERSEC_KIND = "rank"
FUSION_ALPHA = 0.5


def yes_cols(ps, n, fps):
    """per-frame features of the per-second highlight question P(yes).
    rank: within-clip percentile of the second's P(yes) (scale-free: the test videos answer yes far
    more decisively than the training clips); all3: value, value - clip max, clip z-score"""
    v = np.array(ps["yes"], float)
    x = np.interp(np.arange(n) / fps, np.array(ps["t"], float), v)
    if PERSEC_KIND == "rank":
        return np.column_stack([np.searchsorted(np.sort(v), x, side="right") / len(v)])
    return np.column_stack([x, x - v.max(), (x - v.mean()) / (v.std() + 1e-6)])


def feats(r, n, fps, r2=None, ps=None, alpha=None):
    """posterior features of one clip; with a second view r2 the two feature matrices are weighted by alpha;
    with per-second answers ps their columns are appended"""
    if alpha is None:
        alpha = FUSION_ALPHA
    X = features(r, n, fps, use_pos=False)
    if r2 is not None and r2.get("logp"):
        X = (1.0 - alpha) * X + alpha * features(r2, n, fps, use_pos=False)
    if ps is not None:
        X = np.column_stack([X, yes_cols(ps, n, fps)])
    return X


def fit(train_post, train_post2="", train_persec="", alpha=0.5):
    gt = load_gt()
    post = load_post(train_post)
    post2 = load_post(train_post2) if train_post2 else None
    persec = load(train_persec) if train_persec else None
    Xs, ys = [], []
    for v, r in post.items():
        if not r.get("logp") or v not in gt:
            continue
        n, fps = r["n"], r["fps"]
        X = feats(r, n, fps, post2.get(v) if post2 else None, persec.get(v) if persec else None, alpha=alpha)
        y = gt_mask(gt[v], n).astype(float)
        sub = np.arange(0, n, max(1, int(fps / 4)))
        Xs.append(X[sub])
        ys.append(y[sub])
    Xtr, ytr = np.vstack(Xs), np.concatenate(ys)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    w = logit_fit((Xtr - mu) / sd, ytr)
    return w, mu, sd, len(Xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/submission_vlm_mbr169.jsonl")
    ap.add_argument("--train-post", default="scratch/post/train_big.jsonl")
    ap.add_argument("--test-post", default="scratch/post/test_fps25.jsonl")
    ap.add_argument("--full169", default="scratch/full169_p12.jsonl")
    ap.add_argument("--full916", default="outputs/submission_vlm_916_longcut.jsonl")
    ap.add_argument("--video-dir", default="data/test/video")
    ap.add_argument("--ratio", default="169", choices=["169", "916", "both"])
    ap.add_argument("--min-dur", type=float, default=0.0, help="only videos longer than this (s)")
    ap.add_argument("--max-dur", type=float, default=1e9, help="only videos up to this length (s)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--train-post2", default="", help="second-view training posteriors (features averaged)")
    ap.add_argument("--test-post2", default="", help="second-view test posteriors (features averaged)")
    ap.add_argument("--train-persec", default="", help="per-second question answers, training clips")
    ap.add_argument("--test-persec", default="", help="per-second question answers, test videos")
    ap.add_argument("--persec-kind", default="rank", choices=["rank", "all3"])
    ap.add_argument("--fusion-alpha", type=float, default=0.5, help="weight on post2 in (1-alpha)*post + alpha*post2")
    args = ap.parse_args()
    global PERSEC_KIND, FUSION_ALPHA
    PERSEC_KIND = args.persec_kind
    FUSION_ALPHA = args.fusion_alpha
    w, mu, sd, ntr = fit(args.train_post, args.train_post2, args.train_persec, alpha=args.fusion_alpha)
    print("fitted on %d training clips, weights %s" % (ntr, np.round(w, 3)))
    base, post = load(args.base), load(args.test_post)
    post2 = load(args.test_post2) if args.test_post2 else {}
    persec = load(args.test_persec) if args.test_persec else None
    single = 0
    full = {}
    if args.ratio in ("169", "both"):
        full.update({k: v for k, v in load(args.full169).items() if v["targetRatioWH"][0] == 16})
    if args.ratio in ("916", "both") and not args.stats_only:
        full.update({k: v for k, v in load(args.full916).items() if v["targetRatioWH"][0] == 9})
    out, lens, changed = [], [], 0
    for vid, r in base.items():
        is169 = r["targetRatioWH"][0] == 16
        want = (args.ratio == "both") or (args.ratio == "169" and is169) or (args.ratio == "916" and not is169)
        p = post.get(vid)
        if not want or p is None or not p["logp"]:
            out.append(r); continue
        n, fps, W, H = video_meta(glob.glob(os.path.join(args.video_dir, vid + ".*"))[0])
        if not (args.min_dur < n / fps <= args.max_dur):
            out.append(r); continue
        n_use = label_end(n, fps)
        p2 = post2.get(vid)
        single += bool(args.test_post2) and not (p2 and p2["logp"])
        ps = None
        if persec is not None:
            ps = persec.get(vid)
            if ps is None or len(ps["t"]) < 2:
                out.append(r); continue
        q = logit_pred(w, (feats(p, n, fps, p2, ps, alpha=args.fusion_alpha) - mu) / sd)
        a, b = ef_decode(q, n_use, max(1, int(fps / 4)), args.scale)
        lens.append((b - a + 1) / n)
        if args.stats_only:
            continue
        fb = {pp["frame"]: pp["bboxes"] for pp in full[vid]["predictions"]}
        preds = [{"frame": f, "bboxes": fb[f]} for f in range(a, b + 1)]
        changed += preds != r["predictions"]
        out.append({**r, "predictions": preds})
    print("selected %d videos, mean window length %.3f of the video, changed %d" % (len(lens), np.mean(lens), changed))
    if args.test_post2:
        print("second view missing for %d selected videos (single view used)" % single)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print("wrote %s (%d records)" % (args.out, len(out)))


if __name__ == "__main__":
    main()
