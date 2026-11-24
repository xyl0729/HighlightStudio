"""Index / JSONL / video I/O. Video metadata is read exactly like the official
scripts (OpenCV CAP_PROP_*), because the evaluator uses those numbers for the
frame-range and box-bound checks."""
import json
import os

import cv2

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv")
_PATH_KEYS = ("video", "video_path", "video_file", "file_name", "filename", "video_name", "path")


def _parse_json_text(text):
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    try:
        obj = json.loads(text)
        return [obj] if isinstance(obj, dict) else list(obj)
    except json.JSONDecodeError:
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def read_records(path):
    """Records from a JSON list, a JSONL file, a single JSON object, or a directory
    of such files (e.g. one small .jsonl per video), in sorted file order."""
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.lower().endswith((".jsonl", ".json")))
        out = []
        for f in files:
            out.extend(read_records(os.path.join(path, f)))
        return out
    with open(path, encoding="utf-8-sig") as f:
        return _parse_json_text(f.read())


def load_index(path):
    """Official test_index.json (a JSON list), a JSONL file or a directory of them
    -> list of dicts. All original fields are kept so they can be copied into the
    submission."""
    items = read_records(path)
    for it in items:
        it["video_id"] = str(it["video_id"])
        if "targetRatioWH" not in it:
            it["targetRatioWH"] = [16, 9]
    return items


def target_ratio(item):
    tr = item["targetRatioWH"]
    return float(tr[0]), float(tr[1])


def resolve_video_path(item, video_dir):
    """Path given by the index if any, else <video_dir>/<video_id>.<ext>."""
    for k in _PATH_KEYS:
        v = item.get(k)
        if isinstance(v, str) and v:
            p = v if os.path.isabs(v) else os.path.join(video_dir, v)
            if os.path.exists(p):
                return p
    vid = item["video_id"]
    for ext in VIDEO_EXTS:
        p = os.path.join(video_dir, vid + ext)
        if os.path.exists(p):
            return p
    return None


def video_meta(video_path):
    """(n_frames, fps, W, H) exactly as the official scripts read them."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0.0, 0, 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, float(fps), w, h


def iter_frames(video_path, frame_ids):
    """Yield (frame_id, BGR frame) for the requested ids with one sequential pass
    (no seeking, which is unreliable on many codecs). Stops early if decoding
    ends before the metadata frame count."""
    want = sorted(set(int(f) for f in frame_ids if f >= 0))
    if not want:
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    try:
        k, idx = 0, 0
        while k < len(want):
            if not cap.grab():
                break
            if idx == want[k]:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield idx, frame
                k += 1
            idx += 1
    finally:
        cap.release()


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
