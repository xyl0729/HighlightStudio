"""Per-video processing: decode the sampled frames once, choose the highlight
frames, estimate the subject track along the crop's free axis and emit one
fixed-size crop box per kept frame."""
import logging
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import temporal as T
from .geometry import derived_height, free_axis, place_box, safe_crop_width
from .io_utils import iter_frames, video_meta
from .sanitize import sanitize_predictions
from .spatial import PERSON, motion_anchor, smooth_track, subject_anchor

log = logging.getLogger("hlcut")


@dataclass
class Config:
    sample_fps: float = 4.0          # analysis sampling rate
    analysis_width: int = 256        # downscaled width for motion features
    temporal: str = "full"           # full | window | vlm | none
    window_frac: float = 0.6         # kept fraction for temporal=window
    temporal_weights: dict = field(default_factory=lambda: {"motion": 1.0, "subject": 1.0,
                                                            "sharpness": 0.3})
    spatial: str = "detect"          # center | motion | detect | agree | blend | motionfirst
    agree_gate: float = 0.2          # subject-vs-motion disagreement gate (fraction of frame span)
    det_batch: int = 16
    smooth_sec: float = 0.4          # gaussian sigma of the centre trajectory
    headroom: float = 0.1            # crop headroom above a person taller than the crop
    vlm_mode: str = "generic"        # generic query | caption (describe, then ground)
    vlm_fps: float = 2.0
    vlm_max_frames: int = 128
    vlm_max_pixels: int = 256 * 448
    vlm_token_budget: int = 3000
    vlm_min_sec: float = 3.0          # widen short VLM answers to >= max(min_sec, min_frac * duration)
    vlm_min_frac: float = 0.25
    vlm_fallback: str = "full"       # temporal strategy when the VLM gives no usable segment
    # --- options below are off by default (defaults reproduce the 50.15 submission) ---
    y_head: bool = True              # y axis: align a tall person's head (False: box centre)
    shrink_x: float = 1.0            # pull the x track towards the frame centre (1 = off, 0 = centred)
    shrink_y: float = 1.0            # same for the y track
    vlm_center_shrink: float = 1.0   # pull the VLM clip centre towards the video middle (1 = off)
    tail_guard: bool = False         # skip the clip end that the labels can never reach
    keep_dets: bool = False          # keep per-sample detections (scaled to source px) in the recs


def _analyse(path, n_frames, fps, W, H, axis, crop_len, cfg, detector):
    """One sequential pass over sampled frames -> per-sample feature records."""
    step = max(1, int(round(fps / cfg.sample_fps))) if fps > 0 else 1
    recs, batch, prev_gray = [], [], None
    use_det = detector is not None

    def run_batch():
        dets = detector([b[1] for b in batch])
        for (rec, _frame), d in zip(batch, dets):
            sx, sy = W / float(rec["fw"]), H / float(rec["fh"])
            if len(d):
                d = d.copy()
                d[:, [0, 2]] *= sx
                d[:, [1, 3]] *= sy
            if cfg.keep_dets:
                rec["dets"] = d.tolist()
            a, wgt = subject_anchor(d, axis, W, H, crop_len, cfg.headroom,
                                    cfg.y_head) if axis else (None, 0.0)
            rec["s_anchor"], rec["s_weight"] = a, wgt
            persons = d[d[:, 5] == PERSON] if len(d) else d
            rec["subject"] = float(((persons[:, 2] - persons[:, 0]) * (persons[:, 3] - persons[:, 1])
                                    * persons[:, 4]).sum() / (W * H)) if len(persons) else 0.0
        batch.clear()

    for fid, frame in iter_frames(path, range(0, n_frames, step)):
        fh, fw = frame.shape[:2]
        s = cfg.analysis_width / float(fw)
        gray = cv2.cvtColor(cv2.resize(frame, (cfg.analysis_width, max(8, int(round(fh * s)))),
                                       interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        rec = {"fid": fid, "fw": fw, "fh": fh, "s_anchor": None, "s_weight": 0.0, "subject": 0.0}
        rec["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if prev_gray is not None and prev_gray.shape == gray.shape:
            rec["motion"] = float(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)).mean())
            scale = (W / float(gray.shape[1])) if axis == "x" else (H / float(gray.shape[0]))
            rec["m_anchor"], rec["m_frac"] = motion_anchor(prev_gray, gray, axis or "x", scale)
        else:
            rec["motion"], rec["m_anchor"], rec["m_frac"] = 0.0, None, 0.0
        prev_gray = gray
        recs.append(rec)
        if use_det:
            batch.append((rec, frame))
            if len(batch) >= cfg.det_batch:
                run_batch()
    if use_det and batch:
        run_batch()
    if recs and (recs[0]["fw"], recs[0]["fh"]) != (W, H):
        log.warning("decoded frame size %dx%d differs from metadata %dx%d (rotation?)",
                    recs[0]["fw"], recs[0]["fh"], W, H)
    if len(recs) > 1:
        recs[0]["motion"] = recs[1]["motion"]
    return recs


def _segments(recs, n_frames, cfg):
    """Highlight segments over frames [0, n_frames); n_frames may be the tail-guarded length."""
    if cfg.temporal == "none":
        return []
    if cfg.temporal == "full" or not recs:
        return T.full_video(n_frames)
    if cfg.temporal == "window":
        ids = np.array([r["fid"] for r in recs])
        feats = {k: np.array([r[k] for r in recs]) for k in ("motion", "subject", "sharpness")}
        scores = T.highlight_scores(feats, cfg.temporal_weights)
        return T.best_window(ids, scores, n_frames, cfg.window_frac)
    raise ValueError("unknown temporal strategy: %s" % cfg.temporal)


def _anchor_track(recs, n_frames, fps, axis, W, H, cfg):
    """Per-frame anchor coordinate along `axis` (length n_frames).

    spatial=detect  : subject track when it covers enough samples, else the
                      (noisier, so more strongly smoothed) motion centroid, else centre
    spatial=motionfirst : motion centroid first, subject only as a fallback
    spatial=agree   : fuse per sample -- when the subject and motion anchors both
                      exist and disagree by more than agree_gate of the frame span,
                      trust the motion centroid (a large but irrelevant detection
                      stops dragging the box away from what actually moves)
    spatial=blend   : average of subject and motion wherever both exist
    """
    centre = (W if axis == "x" else H) / 2.0
    if cfg.spatial == "center" or not recs:
        return np.full(n_frames, centre)
    span = float(W if axis == "x" else H)
    subj = [(r["fid"], r["s_anchor"]) for r in recs if r["s_anchor"] is not None]
    mot = [(r["fid"], r["m_anchor"]) for r in recs if r.get("m_anchor") is not None]
    enough = max(1, int(0.2 * len(recs)))

    def to_track(pts, sigma_sub):
        ids, vals = zip(*pts)
        track = np.interp(np.arange(n_frames), ids, vals)
        return smooth_track(track, sigma_sub * fps)

    if cfg.spatial in ("detect", "motionfirst"):
        subject_first = cfg.spatial == "detect"
        primary, secondary = (subj, mot) if subject_first else (mot, subj)
        sig_primary = cfg.smooth_sec if subject_first else 2 * cfg.smooth_sec
        sig_secondary = 2 * cfg.smooth_sec if subject_first else cfg.smooth_sec
        if len(primary) >= enough:
            return to_track(primary, sig_primary)
        if len(secondary) >= enough:
            return to_track(secondary, sig_secondary)
        if primary:
            return to_track(primary, sig_primary)
        return np.full(n_frames, centre)

    # agree / blend: fuse the two anchors sample-wise, then interpolate + smooth
    gate = cfg.agree_gate * span
    subj_ids = {fid for fid, _ in subj}
    mot_by_id = dict(mot)
    pts = []
    for fid, s in subj:
        m = mot_by_id.get(fid)
        if m is None:
            pts.append((fid, s))
        elif cfg.spatial == "blend":
            pts.append((fid, (s + m) / 2.0))
        elif abs(s - m) <= gate:
            pts.append((fid, s))
        else:
            pts.append((fid, m))
    for fid, m in mot:
        if fid not in subj_ids:
            pts.append((fid, m))
    pts.sort()
    if len(pts) >= enough:
        return to_track(pts, cfg.smooth_sec)
    return np.full(n_frames, centre)


def _vlm_segments(vlm, path, n_frames, fps, cfg, info, cached=None, n_use=None):
    """Highlight clip from the video LLM, in frames; [] if nothing usable.
    `cached` is a raw answer dict from an earlier run ({"grounding": ...}); it skips the model.
    The clip is kept inside frames [0, n_use) (n_use < n_frames with the tail guard); its
    minimum length is still measured on the whole video."""
    n_use = n_frames if n_use is None else n_use
    if cached is not None:
        from .vlm import answer_segments
        raw = cached
        segs_sec = answer_segments(raw.get("grounding"), n_frames / fps)
    else:
        if vlm is None:
            return []
        frames, ids = vlm.sample_frames(path, n_frames, fps, cfg.vlm_fps, cfg.vlm_max_frames,
                                         cfg.vlm_max_pixels, cfg.vlm_token_budget)
        if len(frames) < 4:
            return []
        segs_sec, raw = vlm.highlight_segments(frames, ids, fps, n_frames, mode=cfg.vlm_mode)
    info["vlm"] = raw
    out = []
    for s, e in segs_sec[:1]:          # the task has a single highlight clip per video
        a, b = s * fps, e * fps
        if cfg.vlm_center_shrink != 1.0:  # shrink the answer's centre towards the middle
            d = (cfg.vlm_center_shrink - 1.0) * ((a + b) / 2.0 - (n_use - 1) / 2.0)
            a, b = a + d, b + d
        f0, f1 = _widen(a, b, n_use, max(cfg.vlm_min_sec * fps, cfg.vlm_min_frac * n_frames))
        if f1 > f0:
            out.append((f0, f1))
    return out


def label_end(n_frames, fps, margin=1):
    """Number of leading frames the labels can reach (tail guard).

    The label teacher scores whole seconds t (kept while t <= end - 0.25 s, end = time of
    the last frame) plus a zero score at the end, and a frame is labelled only where the
    interpolated score is >= 90% of the clip maximum. So nothing after
    t_last + 0.1 * (end - t_last) is ever labelled (checked on the 987 training labels).
    When the end falls within 0.035 s below a whole second the teacher scores it for
    real and there is no dead tail: the whole clip is returned."""
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


def process_video(path, tw, th, cfg, detector=None, vlm=None, vlm_cached=None, memo=None):
    """Returns (predictions, info). Never raises for a readable video: any failure
    after the metadata step degrades to a centred crop over the whole video.
    `memo` (a dict) keeps the per-video analysis pass between calls, so parameter
    sweeps decode and detect each video only once."""
    t0 = time.time()
    n_frames, fps, W, H = video_meta(path)
    info = {"n_frames": n_frames, "fps": fps, "W": W, "H": H}
    if n_frames <= 0 or W <= 0 or H <= 0:
        info["error"] = "unreadable video"
        return [], info
    cw = safe_crop_width(W, H, tw, th)
    ch = derived_height(cw, tw, th)
    axis = free_axis(W, H, cw, tw, th)
    crop_len = cw if axis == "x" else ch
    info.update(crop_w=cw, axis=axis)
    try:
        need_det = detector is not None and cfg.spatial in ("detect", "agree", "blend", "motionfirst") and (
            axis is not None or (cfg.temporal == "window" and cfg.temporal_weights.get("subject")))
        need_pass = axis is not None or cfg.temporal == "window" or (
            cfg.temporal == "vlm" and cfg.vlm_fallback == "window")
        key = (path, axis, crop_len, need_det, cfg.sample_fps, cfg.analysis_width, cfg.y_head)
        if memo is not None and need_pass and key in memo:
            recs = memo[key]
        else:
            recs = _analyse(path, n_frames, fps, W, H, axis, crop_len, cfg,
                            detector if need_det else None) if need_pass else []
            if memo is not None and need_pass:
                memo[key] = recs
        n_use = label_end(n_frames, fps) if cfg.tail_guard else n_frames
        if cfg.temporal == "vlm":
            segs = _vlm_segments(vlm, path, n_frames, fps, cfg, info, vlm_cached, n_use)
            if not segs:
                info["vlm_fallback"] = True
                segs = _segments(recs, n_use, Config(**{**cfg.__dict__, "temporal": cfg.vlm_fallback}))
        else:
            segs = _segments(recs, n_use, cfg)
        track = _anchor_track(recs, n_frames, fps, axis, W, H, cfg) if axis else None
        shrink = cfg.shrink_x if axis == "x" else cfg.shrink_y
        if track is not None and shrink != 1.0:  # the label teacher's centres lean to the middle
            centre = (W if axis == "x" else H) / 2.0
            track = centre + shrink * (track - centre)
        info.update(samples=len(recs), segments=segs)
    except Exception as ex:  # keep the batch alive; fall back to a centred crop
        log.exception("analysis failed for %s, using centred fallback", path)
        info["error"] = repr(ex)
        segs, track = T.full_video(n_frames), None
    preds = []
    for s, e in segs:
        for f in range(s, e + 1):
            if axis == "x":
                cx, cy = track[f], H / 2.0
            elif axis == "y":
                cx, cy = W / 2.0, track[f]
            else:
                cx, cy = W / 2.0, H / 2.0
            preds.append({"frame": f, "bboxes": place_box(cx, cy, W, H, cw, tw, th)})
    preds, fixes = sanitize_predictions(preds, n_frames, W, H, tw, th)
    if fixes:
        info["sanitize"] = dict(fixes)
    info["sec"] = round(time.time() - t0, 2)
    return preds, info
