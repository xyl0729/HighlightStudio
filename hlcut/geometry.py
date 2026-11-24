"""Crop-box geometry shared by the pipeline, the sanitizer and the evaluator.

The crop SIZE is fixed by the task: the largest target-ratio rectangle inside
the source frame (same rule as the official baseline). Only the position of the
box (and which frames are kept) has to be predicted.
"""
import math
from fractions import Fraction


def official_crop_size(W, H, tw, th):
    """Exact copy of compute_crop_size() from the official baseline kit."""
    if tw <= 0 or th <= 0 or W <= 0 or H <= 0:
        return W, H
    target = float(tw) / float(th)
    if W / float(H) >= target:          # source wider than target -> full height
        ch = H
        cw = min(int(round(H * target)), W)
    else:                               # source taller than target -> full width
        cw = W
        ch = min(int(round(W / target)), H)
    return max(1, cw), max(1, ch)


def derived_height(w, tw, th):
    """Height the evaluator derives from a submitted width: h = w * th / tw."""
    return w * float(th) / float(tw)


def safe_crop_width(W, H, tw, th):
    """Official crop width, shrunk by a pixel or two only when the derived height
    would overflow the frame (e.g. 1920x1080 -> 9:16 gives w=608, h=1080.9).
    Guarantees that a box placed inside the frame can never be judged out of bounds."""
    cw, _ = official_crop_size(W, H, tw, th)
    cw = min(int(cw), int(W))
    ratio = Fraction(th) / Fraction(tw)
    while cw > 1 and Fraction(cw) * ratio > H:
        cw -= 1
    return max(1, cw)


def free_axis(W, H, cw, tw, th):
    """Which coordinate the crop can move along: 'x', 'y' or None (box == frame)."""
    ch = derived_height(cw, tw, th)
    if cw < W:
        return "x"
    if ch < H - 1e-6:
        return "y"
    return None


def box_limits(W, H, cw, tw, th):
    """Largest integer x / y that keep the box inside the frame."""
    ch = derived_height(cw, tw, th)
    return max(0, int(W - cw)), max(0, int(math.floor(H - ch + 1e-9)))


def place_box(cx, cy, W, H, cw, tw, th):
    """Integer box [x, y, w] centred as close as possible to (cx, cy) in pixels,
    clamped so the whole box stays inside the frame."""
    ch = derived_height(cw, tw, th)
    x_max, y_max = box_limits(W, H, cw, tw, th)
    x = int(round(cx - cw / 2.0))
    y = int(round(cy - ch / 2.0))
    return [min(max(x, 0), x_max), min(max(y, 0), y_max), int(cw)]


def to_xywh(box, tw, th):
    """[x, y, w] (submission triplet) or [x, y, w, h] -> [x, y, w, h] floats."""
    if len(box) >= 4:
        return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    return [float(box[0]), float(box[1]), float(box[2]), derived_height(float(box[2]), tw, th)]


def iou_xywh(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0
