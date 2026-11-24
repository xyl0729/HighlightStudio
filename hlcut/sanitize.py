"""Make a prediction list valid under the strict checks in metric.check_prediction:
drop unusable entries, clamp boxes inside the frame, de-duplicate and sort."""
import math
from collections import Counter

from .geometry import box_limits, safe_crop_width


def sanitize_predictions(preds, n_frames, W, H, tw, th):
    stats = Counter()
    if n_frames <= 0 or W <= 0 or H <= 0:
        stats["dropped_unreadable_video"] += len(preds)
        return [], stats
    cw_max = safe_crop_width(W, H, tw, th)
    out, seen = [], set()
    for p in preds:
        try:
            f = p["frame"]
            x, y, w = (float(v) for v in p["bboxes"][:3])
            if len(p["bboxes"]) != 3:
                stats["fixed_bbox_length"] += 1
        except Exception:
            stats["dropped_malformed"] += 1
            continue
        if not all(math.isfinite(v) for v in (x, y, w)) or not isinstance(f, (int, float)) \
                or isinstance(f, bool) or not math.isfinite(f):
            stats["dropped_nan"] += 1
            continue
        f = int(round(f))
        if f < 0 or f >= n_frames:
            stats["dropped_frame_out_of_range"] += 1
            continue
        if f in seen:
            stats["dropped_duplicate"] += 1
            continue
        w_i = min(max(int(round(w)), 1), cw_max)
        x_max, y_max = box_limits(W, H, w_i, tw, th)
        x_i = min(max(int(round(x)), 0), x_max)
        y_i = min(max(int(round(y)), 0), y_max)
        if (x_i, y_i, w_i) != (x, y, w):
            stats["clamped_or_rounded"] += 1
        seen.add(f)
        out.append({"frame": f, "bboxes": [x_i, y_i, w_i]})
    out.sort(key=lambda r: r["frame"])
    return out, stats
