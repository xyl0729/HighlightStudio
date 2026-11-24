"""Temporal window rules on the training clips (all 9:16), scored against the teacher
segments (frames, inclusive). Temporal-only F = 2|P & G| / (|P| + |G|).

  python scratch/post/eval_big.py scratch/post/train_big.jsonl
"""
import glob, json, math, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from hlcut.io_utils import video_meta
from hlcut.pipeline import label_end, _widen
from hlcut.vlm import answer_segments

LABELS = os.path.join(ROOT, "data", "train", "高光剪辑训练集", "qvhighlights-videos", "train.jsonl")


def load_gt():
    gt = {}
    for l in open(LABELS, encoding="utf-8-sig"):
        r = json.loads(l)
        gt[r["video_id"]] = [(int(s["start_frame"]), int(s["end_frame"])) for s in r["segments"]]
    return gt


def load_post(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if "n" not in r:
            p = glob.glob(os.path.join(ROOT, "data", "train", "video", r["video_id"] + ".*"))
            if not p:
                continue
            n, fps, _, _ = video_meta(p[0])
            r["n"], r["fps"] = n, fps
        out[r["video_id"]] = r
    return out


def gt_mask(segs, n):
    m = np.zeros(n, bool)
    for a, b in segs:
        m[max(0, a):min(n, b + 1)] = True
    return m


def f_of(a, b, g):
    """F of predicted frames [a, b] against boolean GT mask g."""
    if b < a:
        return 0.0
    hit = g[a:b + 1].sum()
    return 2.0 * hit / ((b - a + 1) + g.sum())


def windows_from_post(r, n_use, frac, hyp_frac, temp, min_sec=3.0):
    """pooled {pred window: p} and {hypothesis window: p} for one clip."""
    fps, n = r["fps"], r["n"]
    lp = np.array(r["logp"]) / temp
    p = np.exp(lp - lp.max()); p /= p.sum()
    W, H = {}, {}
    for q, S, E in zip(p, r["S"], r["E"]):
        a, b = S * fps, E * fps
        w = _widen(a, b, n_use, max(min_sec * fps, frac * n))
        W[w] = W.get(w, 0.0) + q
        if hyp_frac is None:
            h = w
        elif hyp_frac > 0:
            h = _widen(a, b, n_use, hyp_frac * n)
        else:
            h = (int(round(a)), min(n_use - 1, int(round(b))))
        H[h] = H.get(h, 0.0) + q
    return W, H


def mbr(W, H, top=60):
    Hs = list(H.items())
    best, bw = -1.0, None
    for (a0, a1), _ in sorted(W.items(), key=lambda kv: -kv[1])[:top]:
        ef = sum(q * 2.0 * max(0, min(a1, b1) - max(a0, b0) + 1) / ((a1 - a0 + 1) + (b1 - b0 + 1)) for (b0, b1), q in Hs)
        if ef > best:
            best, bw = ef, (a0, a1)
    return bw


def main():
    gt = load_gt()
    post = load_post(sys.argv[1])
    vids = [v for v in post if v in gt and post[v]["logp"]]
    print("clips with posterior:", len(vids))
    res = {}

    def add(name, v, val):
        res.setdefault(name, {})[v] = val

    for v in vids:
        r = post[v]
        n, fps = r["n"], r["fps"]
        n_use = label_end(n, fps)
        g = gt_mask(gt[v], n)
        add("full", v, f_of(0, n - 1, g))
        add("full_tail", v, f_of(0, n_use - 1, g))
        segs = answer_segments(r["greedy"], n / fps)
        s, e = segs[0] if segs else (0.0, n / fps)
        for frac in (0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
            a, b = _widen(s * fps, e * fps, n_use, max(3 * fps, frac * n))
            add("greedy f%.2f" % frac, v, f_of(a, b, g))
            for temp in (1.0, 2.0):
                for hf in (None, 0.0):
                    W, H = windows_from_post(r, n_use, frac, hf, temp)
                    a, b = mbr(W, H)
                    add("mbr f%.2f T%.0f hf%s" % (frac, temp, hf), v, f_of(a, b, g))
    base = res["full_tail"]
    for name, d in res.items():
        x = np.array([d[v] for v in vids])
        dlt = x - np.array([base[v] for v in vids])
        extra = ""
        if name.startswith("mbr"):
            gk = "greedy " + name.split()[1]
            dg = x - np.array([res[gk][v] for v in vids])
            extra = "   vs %s %+.4f (se %.4f)" % (gk, dg.mean(), dg.std() / math.sqrt(len(x)))
        print("%-26s F %.4f   vs full_tail %+.4f (se %.4f)%s" % (name, x.mean(), dlt.mean(),
                                                               dlt.std() / math.sqrt(len(x)), extra))


if __name__ == "__main__":
    main()
