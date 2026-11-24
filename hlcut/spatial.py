"""Subject position along the free axis of the fixed-size crop.

Per sampled frame we estimate the pixel coordinate the crop should be centred on
(x for landscape->portrait, y for portrait->landscape); the pipeline then
interpolates to every frame, smooths the trajectory and clamps the box.
"""
import numpy as np

PERSON = 0
# COCO classes that can plausibly be the main subject of a highlight clip
SUBJECT_CLASS_WEIGHT = {
    0: 1.0,                                   # person
    1: 0.6, 2: 0.5, 3: 0.6, 5: 0.4, 7: 0.4,   # bicycle car motorcycle bus truck
    14: 0.5, 15: 0.6, 16: 0.6, 17: 0.6,       # bird cat dog horse
    18: 0.5, 19: 0.5, 20: 0.5, 21: 0.5, 22: 0.5, 23: 0.5,  # sheep cow elephant bear zebra giraffe
    32: 0.5,                                  # sports ball
    36: 0.5, 30: 0.5, 31: 0.5, 37: 0.5,       # skateboard skis snowboard surfboard
}


class YoloDetector:
    """Thin wrapper around an Ultralytics YOLO model."""

    def __init__(self, weights="yolo11s.pt", device=None, conf=0.25, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def __call__(self, frames_bgr):
        """list of BGR frames -> list of (N, 6) arrays [x1, y1, x2, y2, conf, cls]."""
        if not frames_bgr:
            return []
        res = self.model.predict(frames_bgr, conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)
        out = []
        for r in res:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 6), dtype=np.float32))
                continue
            out.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                       b.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32))
        return out


def subject_anchor(dets, axis, W, H, crop_len, headroom=0.1, head_align=True):
    """Pixel coordinate along `axis` for the main subject in one frame, and a
    confidence weight; (None, 0) when nothing usable is detected.

    Detections are weighted by (class weight * conf * area)^2 so the largest,
    most confident subject dominates while a group still pulls the crop towards
    its centre. When a person is taller than a vertical-moving crop, the crop is
    aligned to the head (with some headroom) instead of the body centre, unless
    head_align is False (then the box centre is used, like the label teacher does)."""
    if dets is None or len(dets) == 0:
        return None, 0.0
    num, den = 0.0, 0.0
    for x1, y1, x2, y2, conf, cls in dets:
        cw_ = SUBJECT_CLASS_WEIGHT.get(int(cls), 0.0)
        if cw_ <= 0:
            continue
        area = max(0.0, (x2 - x1) * (y2 - y1)) / float(W * H)
        wgt = (cw_ * conf * area) ** 2
        if wgt <= 0:
            continue
        if axis == "x":
            a = 0.5 * (x1 + x2)
        else:
            h = y2 - y1
            if head_align and int(cls) == PERSON and h > 0.9 * crop_len:
                a = y1 + crop_len * (0.5 - headroom)
            else:
                a = 0.5 * (y1 + y2)
        num += wgt * a
        den += wgt
    if den <= 0:
        return None, 0.0
    return num / den, den


def motion_anchor(prev_gray, gray, axis, scale, thresh=12):
    """Centroid of frame-difference energy along `axis` (in source pixels) and
    the fraction of changed pixels; (None, 0) when there is almost no motion."""
    if prev_gray is None:
        return None, 0.0
    diff = np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16))
    mask = diff > thresh
    frac = float(mask.mean())
    # Keep compact movers that occupy only a small fraction of a wide shot.
    # A 2x2 opening removes isolated codec noise before using that lower floor.
    if frac < 0.002:
        import cv2

        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        frac = float(mask.mean())
    if int(mask.sum()) < max(4, int(mask.size * 0.0002)) or frac > 0.35:
        return None, frac
    prof = mask.sum(axis=0 if axis == "x" else 1).astype(np.float64)
    pos = np.arange(prof.size) + 0.5
    return float((prof * pos).sum() / prof.sum()) * scale, frac


def smooth_track(values, sigma):
    """Gaussian smoothing with edge padding (sigma in samples)."""
    v = np.asarray(values, dtype=np.float64)
    if sigma <= 0 or v.size < 3:
        return v
    r = int(max(1, round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(v, r, mode="edge")
    return np.convolve(padded, k, mode="valid")
