"""Pure temporal probability decoder utilities used by the inference pipeline."""
import math
import numpy as np

WIDTHS = (0.3, 0.6)


def post_probs(r, temp=1.0):
    lp = np.array(r["logp"]) / temp
    p = np.exp(lp - lp.max())
    return p / p.sum()


def _widen(a, b, n_frames, min_len):
    """Frame span [a, b] grown symmetrically to >= min_len frames, shifted to stay in range."""
    a, b = float(min(a, b)), float(max(a, b))
    if b - a < min_len:
        c = (a + b) / 2.0
        a, b = c - min_len / 2.0, c + min_len / 2.0
        if a < 0:
            a, b = 0.0, b - a
        if b > n_frames - 1:
            a, b = a - (b - (n_frames - 1)), float(n_frames - 1)
    f0 = max(0, min(n_frames - 1, int(round(a))))
    f1 = max(0, min(n_frames - 1, int(round(b))))
    return f0, f1


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
    return np.column_stack(cols)


def logit_pred(w, X):
    return 1 / (1 + np.exp(-(np.column_stack([X, np.ones(len(X))]) @ w)))


def ef_decode(q, n_use, step, scale=1.0):
    """[a, b] maximizing expected overlap on a frame grid."""
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


def label_end(n_frames, fps, margin=1):
    """Tail frame guard for highlight decoding."""
    if n_frames <= 1 or fps <= 0:
        return n_frames
    end = (n_frames - 1) / float(fps)
    if math.ceil(end) - end < 0.035:
        return n_frames
    t_last = math.floor(end - 0.25)
    if t_last < 0:
        return n_frames
    cut = t_last + 0.1 * (end - t_last)
    return max(1, min(n_frames, int(math.floor(cut * fps)) + 1 + margin))


def temporal_scale(ratio):
    if list(ratio) == [16, 9]:
        return 1.0
    if list(ratio) == [9, 16]:
        return 1.7
    raise ValueError("Unsupported target ratio")


def choose_window(q, n, fps, ratio):
    if n <= 0 or fps <= 0 or len(q) != n or not np.isfinite(q).all():
        raise ValueError("Invalid temporal input")
    return ef_decode(q, label_end(n, fps), max(1, int(fps / 4)), temporal_scale(ratio))
