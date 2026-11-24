"""Temporal selection: which frames go into the highlight clip.

All strategies return a list of inclusive (start_frame, end_frame) segments.
"""
import numpy as np


def zscore(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return v
    sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-9 else np.zeros_like(v)


def full_video(n_frames):
    return [(0, n_frames - 1)] if n_frames > 0 else []


def best_window(sample_ids, scores, n_frames, frac, min_frames=1):
    """Single contiguous window covering `frac` of the video whose mean
    per-sample score is highest. frac >= 1 -> whole video."""
    if n_frames <= 0:
        return []
    length = int(round(max(min_frames, min(1.0, frac) * n_frames)))
    if length >= n_frames or len(sample_ids) < 2:
        return full_video(n_frames)
    # per-frame score by step interpolation of the sampled scores
    per_frame = np.interp(np.arange(n_frames), sample_ids, scores)
    csum = np.concatenate([[0.0], np.cumsum(per_frame)])
    sums = csum[length:] - csum[:-length]
    start = int(np.argmax(sums))
    return [(start, start + length - 1)]


def highlight_scores(features, weights):
    """Weighted sum of z-scored per-sample features; missing features are skipped."""
    total = None
    for name, w in weights.items():
        f = features.get(name)
        if f is None or w == 0:
            continue
        z = w * zscore(f)
        total = z if total is None else total + z
    if total is None:
        n = len(next(iter(features.values()))) if features else 0
        total = np.zeros(n)
    return total
