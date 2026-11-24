"""Expected-F window decode from the answer posterior (training clips, all 9:16).

Per frame (sub-sampled) features from the Video-ORA-4B answer distribution:
  m_raw  P(frame inside the answer [S, E])
  m_w    P(frame inside the answer widened to w * duration), w in WIDTHS
  pos    relative position terms (optional; --no-pos drops them)
A logistic model q(frame in GT | features) is fitted with 5-fold CV by clip, then each
clip's window [a, b] maximises 2 * sum_[a,b] q / ((b - a + 1) + sum q) inside the tail guard.

  python scratch/post/fit_decode.py scratch/post/train_big.jsonl [--no-pos]
"""
import argparse, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_big import load_gt, load_post, gt_mask, f_of, windows_from_post, mbr
from hlcut.pipeline import label_end, _widen
from hlcut.vlm import answer_segments

WIDTHS = (0.3, 0.6)


def post_probs(r, temp=1.0):
    lp = np.array(r["logp"]) / temp
    p = np.exp(lp - lp.max())
    return p / p.sum()


def features(r, n, fps, use_pos=True, temp=1.0):
    """(n, k) feature matrix for every frame of a clip."""
    p = post_probs(r, temp)
    S, E = np.array(r["S"], float) * fps, np.array(r["E"], float) * fps
    f = np.arange(n, dtype=float)
    cols = []
    m = np.zeros(n)
    for q, a, b in zip(p, S, E):
        m[int(a):int(min(n, b + 1))] += q
    cols.append(m)
    for w in WIDTHS:
        m = np.zeros(n)
        for q, a, b in zip(p, S, E):
            a2, b2 = _widen(a, b, n, w * n)
            m[a2:b2 + 1] += q
        cols.append(m)
    c_mean = float((p * (S + E) / 2).sum())
    cols.append(np.abs(f - c_mean) / n)
    if use_pos:
        rel = f / n
        cols += [rel, rel ** 2, (f < fps).astype(float), np.full(n, math.log(n / fps))]
    X = np.column_stack(cols)
    return X


def logit_fit(X, y, lam=1.0, iters=30):
    X1 = np.column_stack([X, np.ones(len(X))])
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        z = X1 @ w
        pr = 1 / (1 + np.exp(-z))
        g = X1.T @ (pr - y) + lam * np.r_[w[:-1], 0]
        Hs = (X1 * (pr * (1 - pr))[:, None]).T @ X1 + lam * np.diag(np.r_[np.ones(len(w) - 1), 0])
        w -= np.linalg.solve(Hs, g)
    return w


def logit_pred(w, X):
    return 1 / (1 + np.exp(-(np.column_stack([X, np.ones(len(X))]) @ w)))


def ef_decode(q, n_use, step, scale=1.0):
    """[a, b] maximising 2 * sum q[a:b+1] / ((b - a + 1) + scale * sum q) on a frame grid."""
    q = q[:n_use]
    cs = np.r_[0, np.cumsum(q)]
    tot = scale * q.sum()
    grid = np.unique(np.r_[np.arange(0, n_use, step), n_use - 1]).astype(int)
    best, bw = -1, (0, n_use - 1)
    for a in grid:
        bs = grid[grid >= a]
        v = 2 * (cs[bs + 1] - cs[a]) / ((bs - a + 1) + tot)
        j = int(np.argmax(v))
        if v[j] > best:
            best, bw = v[j], (int(a), int(bs[j]))
    return bw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("post")
    ap.add_argument("--no-pos", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0)
    args = ap.parse_args()
    gt = load_gt()
    post = load_post(args.post)
    vids = sorted(v for v in post if v in gt and post[v]["logp"])
    rng = np.random.RandomState(0)
    order = rng.permutation(len(vids))
    folds = np.array_split(order, 5)
    data = {}
    for v in vids:
        r = post[v]
        n, fps = r["n"], r["fps"]
        X = features(r, n, fps, not args.no_pos, args.temp)
        y = gt_mask(gt[v], n).astype(float)
        data[v] = (X, y, n, fps)
    res = {k: [] for k in ("full_tail", "greedy60", "mbr60", "mbr40T2", "mbr50T2", "ef", "ef_s0.8", "ef_s1.2")}
    for k in range(5):
        te = [vids[i] for i in folds[k]]
        tr = [vids[i] for j in range(5) if j != k for i in folds[j]]
        Xs, ys = [], []
        for v in tr:
            X, y, n, fps = data[v]
            sub = np.arange(0, n, max(1, int(fps / 4)))
            Xs.append(X[sub]); ys.append(y[sub])
        Xtr, ytr = np.vstack(Xs), np.concatenate(ys)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        w = logit_fit((Xtr - mu) / sd, ytr)
        if k == 0:
            print("weights", np.round(w, 3))
        for v in te:
            X, y, n, fps = data[v]
            g = y.astype(bool)
            n_use = label_end(n, fps)
            q = logit_pred(w, (X - mu) / sd)
            res["full_tail"].append(f_of(0, n_use - 1, g))
            segs = answer_segments(post[v]["greedy"], n / fps)
            s, e = segs[0] if segs else (0, n / fps)
            a, b = _widen(s * fps, e * fps, n_use, max(3 * fps, 0.6 * n))
            res["greedy60"].append(f_of(a, b, g))
            W, H = windows_from_post(post[v], n_use, 0.6, 0.0, 1.0)
            a, b = mbr(W, H)
            res["mbr60"].append(f_of(a, b, g))
            for nm, fr in (("mbr40T2", 0.4), ("mbr50T2", 0.5)):
                W, H = windows_from_post(post[v], n_use, fr, None, 2.0)
                a, b = mbr(W, H)
                res[nm].append(f_of(a, b, g))
            for name, sc in (("ef", 1.0), ("ef_s0.8", 0.8), ("ef_s1.2", 1.2)):
                a, b = ef_decode(q, n_use, max(1, int(fps / 4)), sc)
                res[name].append(f_of(a, b, g))
    base = np.array(res["full_tail"])
    g60 = np.array(res["greedy60"])
    for name, x in res.items():
        x = np.array(x)
        d = x - base
        d2 = x - g60
        print("%-10s F %.4f  vs full_tail %+.4f (se %.4f)  vs greedy60 %+.4f (se %.4f)" % (
            name, x.mean(), d.mean(), d.std() / math.sqrt(len(d)), d2.mean(), d2.std() / math.sqrt(len(d2))))


if __name__ == "__main__":
    main()
