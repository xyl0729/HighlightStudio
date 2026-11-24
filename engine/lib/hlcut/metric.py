"""Local re-implementation of the official metric (spec, section 6).

Per video: P = S_IoU / N_pred, R = S_IoU / N_gt, F = 2PR / (P + R)
(equivalently F = 2 * S_IoU / (N_pred + N_gt)). Invalid, duplicate or
wrong-frame predictions still count in N_pred. N_gt = N_pred = 0 -> F = 1;
exactly one of them 0 -> F = 0. Final score = mean F over indexed videos * 100.
The checks are deliberately strict so a file that passes here also passes the
official evaluator.
"""
import math
from collections import Counter

from .geometry import derived_height, iou_xywh, to_xywh


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_prediction(p, n_frames, W, H, tw, th):
    """(frame, [x, y, w, h]) for a valid prediction, else (None, reason)."""
    if not isinstance(p, dict) or "frame" not in p or "bboxes" not in p:
        return None, "missing_field"
    f = p["frame"]
    if not _is_int(f):
        return None, "frame_not_int"
    if f < 0 or f >= n_frames:
        return None, "frame_out_of_range"
    b = p["bboxes"]
    if not isinstance(b, (list, tuple)) or len(b) != 3 or not all(_is_num(v) for v in b):
        return None, "bbox_malformed"
    x, y, w = float(b[0]), float(b[1]), float(b[2])
    if x < 0 or y < 0 or w <= 0:
        return None, "bbox_negative_or_empty"
    h = derived_height(w, tw, th)
    if x + w > W or y + h > H:
        return None, "bbox_out_of_frame"
    return f, [x, y, w, h]


def evaluate_video(preds, gt, n_frames, W, H, tw, th):
    """preds: submitted list for one video; gt: {frame: [x, y, w, h]}."""
    n_pred, n_gt = len(preds), len(gt)
    seen, invalid = set(), Counter()
    s_iou, n_hit = 0.0, 0
    for p in preds:
        f, box = check_prediction(p, n_frames, W, H, tw, th)
        if f is None:
            invalid[box] += 1
            continue
        if f in seen:
            invalid["duplicate_frame"] += 1
            continue
        seen.add(f)
        g = gt.get(f)
        if g is not None:
            s_iou += iou_xywh(box, g)
            n_hit += 1
    if n_gt == 0 and n_pred == 0:
        P = R = F = 1.0
    elif n_gt == 0 or n_pred == 0:
        P = R = F = 0.0
    else:
        P, R = s_iou / n_pred, s_iou / n_gt
        F = 2 * P * R / (P + R) if P + R > 0 else 0.0
    return {"F": F, "P": P, "R": R, "S_iou": s_iou, "n_pred": n_pred, "n_gt": n_gt,
            "n_hit": n_hit, "mean_iou_hit": s_iou / n_hit if n_hit else 0.0,
            "invalid": dict(invalid)}


def load_gt_records(records):
    """GT JSONL records -> {video_id: {"ratio": (tw, th), "boxes": {frame: [x,y,w,h]}, "meta": {...}}}.
    Accepts the submission layout ("predictions") or "annotations"/"gt"/"labels";
    boxes may be triplets [x, y, w] (h derived) or [x, y, w, h].
    Time-only labels (QVHighlights style "relevant_windows": [[start_s, end_s], ...])
    are kept as "windows"; use gt_frames() to turn them into frame numbers."""
    out = {}
    for r in records:
        vid = str(r.get("video_id", r.get("vid")))
        tw, th = (float(v) for v in r.get("targetRatioWH", [16, 9]))
        items = None
        for k in ("annotations", "gt", "labels", "predictions"):
            if isinstance(r.get(k), list):
                items = r[k]
                break
        boxes = {}
        for it in items or []:
            b = it.get("bboxes", it.get("bbox"))
            if b and isinstance(b[0], (list, tuple)):
                b = b[0]
            boxes[int(it["frame"])] = to_xywh(b, tw, th)
        meta = {k: r[k] for k in ("num_frames", "width", "height", "fps") if k in r}
        windows = r.get("relevant_windows") or r.get("windows")
        if not boxes and isinstance(windows, list):
            windows = [(float(a), float(b)) for a, b in windows]
            if vid in out:  # QVHighlights repeats a video once per query: merge the windows
                windows = out[vid].get("windows", []) + windows
        else:
            windows = None
        out[vid] = {"ratio": (tw, th), "boxes": boxes, "meta": meta, "windows": windows}
    return out


def gt_frames(g, fps, n_frames):
    """Labelled frame numbers of one video: box frames, or frames inside the time windows."""
    if g.get("boxes"):
        return set(g["boxes"])
    frames = set()
    for a, b in g.get("windows") or []:
        f0 = max(0, int(round(a * fps)))
        f1 = min(n_frames - 1, int(round(b * fps)) - 1)
        frames.update(range(f0, f1 + 1))
    return frames


def evaluate_temporal(preds, frames):
    """Frame-selection-only F (every hit counts as IoU 1), for time-only labels."""
    pred = {p["frame"] for p in preds if isinstance(p.get("frame"), int)}
    n_pred, n_gt, hit = len(pred), len(frames), len(pred & frames)
    if n_gt == 0 and n_pred == 0:
        return {"F": 1.0, "P": 1.0, "R": 1.0}
    if n_gt == 0 or n_pred == 0:
        return {"F": 0.0, "P": 0.0, "R": 0.0}
    P, R = hit / n_pred, hit / n_gt
    return {"F": 2 * P * R / (P + R) if hit else 0.0, "P": P, "R": R}


def evaluate(pred_records, gt, video_ids, meta_fn):
    """pred_records: submission lines; gt: output of load_gt_records;
    video_ids: videos to score (normally the whole index);
    meta_fn(video_id) -> (n_frames, W, H). Returns (score_0_100, per_video)."""
    preds_by_vid, dup_lines = {}, 0
    for r in pred_records:
        vid = str(r.get("video_id"))
        if vid in preds_by_vid:
            dup_lines += 1
            continue
        preds_by_vid[vid] = r.get("predictions") or []
    per_video = {}
    for vid in video_ids:
        g = gt.get(vid, {"ratio": (16.0, 9.0), "boxes": {}})
        tw, th = g["ratio"]
        n_frames, W, H = meta_fn(vid)
        per_video[vid] = evaluate_video(preds_by_vid.get(vid, []), g["boxes"], n_frames, W, H, tw, th)
    score = 100.0 * sum(v["F"] for v in per_video.values()) / max(1, len(per_video))
    return score, per_video, dup_lines
