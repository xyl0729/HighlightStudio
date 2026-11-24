import numpy as np
import math

def native_edges(start, end, fps):
    if not 0 <= start < end or not math.isfinite(fps) or fps <= 0:
        raise ValueError('positive native half-open clock required')
    step = max(1, math.ceil(0.25 * fps - 1e-12))
    e = np.r_[np.arange(start, end, step, dtype=np.int64), end]
    if len(e) > 2 and e[-1] - e[-2] < step:
        e = np.delete(e, -2)
    return e

def runs(mask):
    d = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]

def proposal_features(raw, q, base, keep, fps, scale):
    raw = np.asarray(raw, float)
    q = np.asarray(q, float)
    base = np.asarray(base, bool)
    keep = np.asarray(keep, bool)
    n = len(q)
    if raw.shape != (n, 4) or base.shape != (n,) or keep.shape != (n,) or (not base.any()) or (not keep.any()) or np.any(keep & ~base):
        raise ValueError('finite native evidence and nonempty anchor subset required')
    x = np.column_stack([raw, q])
    removed = base & ~keep

    def stats(m):
        return (x[m].mean(0), x[m].std(0), x[m].max(0)) if m.any() else (np.zeros(5),) * 3
    bm, bs, _ = stats(base)
    km, ks, kmax = stats(keep)
    rm, rs, rmax = stats(removed)
    bl = float(base.sum())
    kl = float(keep.sum())
    mass = float(q.sum() * scale)
    bf = 2 * float(q[base].sum()) / (bl + mass)
    kf = 2 * float(q[keep].sum()) / (kl + mass)
    rr = runs(keep)
    gaps = runs(removed)
    gap = max((b - a for a, b in gaps), default=0)
    geo = [math.log1p(n / fps), math.log1p(bl / fps), bl / n, kl / bl, 1 - kl / bl, len(rr) / 8, gap / bl, removed.sum() / bl, mass / n, bf, kf, kf - bf, scale / 1.7, bm[-1], km[-1], rm[-1]]
    out = np.r_[bm, bs, km, ks, rm, rs, kmax, rmax, geo]
    if out.shape != (56,) or not np.isfinite(out).all():
        raise ValueError('finite56 feature contract')
    return out

def gap(q, a, b, fps, scale):
    q = np.asarray(q, float)
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)) or (not 0 <= a < b <= len(q)) or (not math.isfinite(scale)) or (scale <= 0):
        raise ValueError('finite native probabilities, window and positive mass scale required')
    minimum = max(1, math.ceil(0.25 * fps - 1e-12))
    if b - a < 3 * minimum:
        return None
    edges = native_edges(a, b, fps)
    prefix = np.r_[0.0, np.cumsum(q)]
    total = prefix[b] - prefix[a]
    mass = scale * float(q.sum())
    best = None
    for l in edges[1:-1]:
        rs = edges[(edges - l >= minimum) & (b - edges >= minimum)]
        if l - a < minimum or not len(rs):
            continue
        values = 2 * (total - (prefix[rs] - prefix[l])) / (b - a - rs + l + mass)
        val = float(values.max())
        r = int(rs[np.flatnonzero(values == val)[0]])
        key = (-val, r - int(l), int(l), r)
        if best is None or key < best:
            best = key
    return None if best is None else (best[2], best[3])
