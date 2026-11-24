"""Run the reusable highlight and motion tracking pipeline for one source video."""
from dataclasses import replace
import math
from pathlib import Path

from hlcut.geometry import free_axis, safe_crop_width
from hlcut.pipeline import Config, process_video
from hlcut.spatial import YoloDetector
from hlcut.temporal import split_motion_gap
from hlcut.vlm import VideoLLM

from .media import Media


class Analyzer:
    def __init__(self, model: Path | None, detector_weights: Path | None = None):
        self.model = model
        self.detector_weights = detector_weights
        self.vlm = None
        self.detector = None

    def analyze(self, source: Path, media: Media, ratio: str, mode: str):
        rw, rh = map(int, ratio.split(":"))
        crop_width = safe_crop_width(media.width, media.height, rw, rh)
        can_reframe = free_axis(media.width, media.height, crop_width, rw, rh) is not None
        detector = None
        warnings = []
        if mode == "ai" and can_reframe:
            if self.detector_weights and self.detector is None:
                if not self.detector_weights.is_file():
                    raise ValueError("HIGHLIGHT_YOLO_WEIGHTS must point to a local YOLO11s weight file.")
                self.detector = YoloDetector(str(self.detector_weights.resolve()), device="0")
            detector = self.detector
            if detector is None:
                warnings.append("未配置主体检测。裁剪使用运动轨迹。")
        config = Config(temporal="full", spatial="agree" if detector else "motion",
                        det_batch=4, smooth_sec=0.28)
        memo = {}
        predictions, info = process_video(str(source), rw, rh, config, detector=detector, memo=memo)
        if info.get("error") or len(predictions) != media.frame_count:
            raise ValueError("Could not compute the crop track for every frame.")
        boxes = [point["bboxes"] for point in predictions]
        tracking = info["crop_tracking"]
        if tracking == "center":
            warnings.append("未找到稳定的主体或运动轨迹，已采用居中构图。")
        elif detector and tracking == "motion":
            warnings.append("主体检测未提供稳定的构图依据，已使用运动轨迹。")
        if mode == "ai":
            if self.vlm is None:
                if self.model is None or not (self.model / "config.json").is_file():
                    raise ValueError("Set HIGHLIGHT_MODEL_DIR to a local Video-ORA-4B model directory.")
                import torch

                if not torch.cuda.is_available():
                    raise ValueError("AI mode needs a CUDA GPU. Quick mode can run on CPU.")
                self.vlm = VideoLLM(str(self.model.resolve()), quant="4bit", device="cuda:0")
            frames, ids = self.vlm.sample_frames(
                str(source), media.frame_count, media.fps, max_frames=64, token_budget=2048,
            )
            if len(frames) >= 4:
                prompt = "highlights" if media.duration >= 8 else "generic"
                spans, _ = self.vlm.highlight_segments(frames, ids, media.fps, media.frame_count, mode=prompt)
                if not spans and prompt == "highlights":
                    spans, _ = self.vlm.highlight_segments(frames, ids, media.fps, media.frame_count, mode="generic")
                # Restore a little context around decisions made from sampled frames,
                # without bridging silence between separate events.
                context = round(0.20 * media.fps)
                selected = []
                for start, end in spans:
                    a = max(0, math.floor(start * media.fps) - context)
                    b = min(media.frame_count, math.ceil(end * media.fps) + context)
                    if b > a:
                        selected.append((a, b))
                merged = []
                for start, end in sorted(selected):
                    if merged and start <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                    else:
                        merged.append((start, end))
                if merged:
                    if media.duration >= 8 and len(merged) == 1:
                        if not memo:
                            process_video(str(source), rw, rh, replace(config, temporal="window"), memo=memo)
                        samples = next(iter(memo.values()), [])
                        split = split_motion_gap(
                            merged, [row["fid"] for row in samples], [row["motion"] for row in samples],
                            media.fps, media.frame_count,
                        )
                        if len(split) > len(merged):
                            warnings.append("高光长窗中检测到连续低运动间隔，已拆成两段。请在时间轴确认。")
                        merged = split
                    return merged[:50], boxes, "video-ora-4b", warnings, tracking
            warnings.append("Video-ORA 未返回有效的时间区间，已使用快速模式的窗口。")
        _, info = process_video(str(source), rw, rh, replace(config, temporal="window"),
                                detector=detector, memo=memo)
        if info.get("error"):
            raise ValueError("Could not select a highlight window.")
        return [(start, end + 1) for start, end in info["segments"]], boxes, "motion", warnings, tracking
