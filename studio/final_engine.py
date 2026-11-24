"""Bridge the bundled multimodal highlight inference engine to the web editor's frame coordinates."""
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from .media import Media

logger = logging.getLogger(__name__)
RUNTIME_ASSETS = ("Video-ORA-4B", "CLIP-ViT-B-32", "Whisper-small", "UNISAL-upstream", "yolo11s.pt")


def project_result(predictions, native: dict, preview: Media, ratio: tuple[int, int]):
    """Keep delivered frame selections and map the track to the preview clock.

    Between delivered boxes, interpolate for browsing and manual edits.
    Selected frames at the native cadence retain their original box.
    """
    if not predictions:
        raise ValueError("推理引擎没有选出有效高光帧。")
    ids = np.asarray([row["frame"] for row in predictions], dtype=int)
    values = np.asarray([row["bboxes"] for row in predictions], dtype=float)
    if (np.any(np.diff(ids) <= 0) or ids[0] < 0 or ids[-1] >= native["n"]
            or values.shape != (len(ids), 3) or not np.isfinite(values).all()):
        raise ValueError("推理引擎返回了无效的逐帧结果。")
    times = np.arange(preview.frame_count) * float(native["fps"]) / preview.fps
    mapped = np.clip(np.floor(times + 1e-7).astype(int), 0, native["n"] - 1)
    kept = np.isin(mapped, ids)
    edges = np.flatnonzero(np.diff(np.r_[False, kept, False]))
    segments = [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2])]
    if not segments:
        raise ValueError("高光时间段没有映射到网页预览。")

    rw, rh = ratio
    sx, sy = preview.width / native["W"], preview.height / native["H"]
    boxes = []
    projected = [np.interp(mapped, ids, col) for col in values.T]
    for x, y, width in zip(*projected):
        w = min(max(1, round(width * sx)), preview.width, math.floor(preview.height * rw / rh))
        boxes.append([
            max(0, min(round(x * sx), preview.width - w)),
            max(0, min(round(y * sy), math.floor(preview.height - w * rh / rw))),
            w,
        ])
    return segments, boxes


class FinalEngine:
    def __init__(self, code: str | None, models: str | None):
        bundled = Path(__file__).resolve().parent.parent / "engine"
        if code:
            self.code = Path(code).resolve()
        elif (bundled / "run_raw.py").is_file():
            self.code = bundled
        else:
            self.code = None
        self.models = Path(models).resolve() if models else None

    @property
    def configured(self) -> bool:
        return bool(self.code and self.models and (self.code / "run_raw.py").is_file()
                    and all((self.models / name).exists() for name in RUNTIME_ASSETS))

    def analyze(self, upload: Path, preview: Media, aspect_ratio: str):
        if not self.configured:
            raise ValueError("请先配置本机推理代码和模型目录（HIGHLIGHT_FINAL_MODELS）。")
        ratio = tuple(map(int, aspect_ratio.split(":")))
        with TemporaryDirectory(prefix="final-", dir=upload.parent) as staging:
            root = Path(staging)
            sources = root / "video"
            sources.mkdir()
            # The unchanged runner's reproducibility preflight requires two
            # index rows. The second ID points to the same newly uploaded bytes.
            ids = ("clip", "preflight")
            for name in ids:
                target = sources / f"{name}.mp4"
                try:
                    os.link(upload, target)
                except OSError:
                    shutil.copyfile(upload, target)
            index = root / "index.json"
            index.write_text(json.dumps([{"video_id": name, "targetRatioWH": ratio} for name in ids]),
                             encoding="utf-8")
            output = root / "run"
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUTF8="1")
            env.pop("PYTHONPATH", None)
            command = [sys.executable, str(self.code / "run_raw.py"), "--index", str(index),
                       "--video-dir", str(sources), "--models", str(self.models), "--work", str(output)]
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=3600)
            if result.returncode != 0:
                record = output / "PROCESS_EXIT.json"
                failure = "启动"
                if record.is_file():
                    stages = json.loads(record.read_text(encoding="utf-8"))["stages"]
                    failure = next((key for key, code in stages.items() if code), failure)
                log = output / f"{failure}.stderr.log"
                logger.error("Inference engine failed at %s: %s", failure,
                             log.read_text(encoding="utf-8", errors="replace")[-3000:] if log.is_file()
                             else result.stderr[-3000:])
                raise ValueError(f"多模态引擎在 {failure} 阶段失败，请查看后端日志。")
            manifest = json.loads((output / "code/raw_manifest.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in (output / "final/predictions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()]
            native = next(row for row in manifest["cases"] if row["video_id"] == "clip")
            selected = next(row for row in rows if row["video_id"] == "clip")
            return project_result(selected["predictions"], native, preview, ratio)
