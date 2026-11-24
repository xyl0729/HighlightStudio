"""Video LLM wrapper (Video-ORA / Qwen3.5 / Qwen3-VL family) for highlight
localization. Frames are decoded with OpenCV and handed to the processor with
explicit VideoMetadata, so the model sees correct "<t seconds>" timestamps and no
video backend (decord/torchcodec) is needed on Windows. Weights can be loaded in
4-bit / 8-bit to fit an 8 GB GPU; the vision tower stays in bf16."""
import re

import cv2
import numpy as np

from .io_utils import iter_frames

# Training-aligned prompts from OraRL eval/task/eval_prompt.py
TEMPORAL_GROUNDING_PROMPT = (
    'To accurately pinpoint the event "{}" in the video, '
    "determine the precise time period of the event. "
    "Provide the start and end times (in seconds) "
    'in the format "start time to end time" within <answer> </answer> tags. '
    "Example: <answer> 12 to 18 </answer>"
)
# Free-form prompts derail this RL-tuned model (it drifts into box lists); the plain
# "What is happening" question was the one that gave clean sentences in smoke tests.
CAPTION_PROMPT = "What is the most exciting moment in this video? Answer briefly."
GENERIC_HIGHLIGHT_QUERY = "the most exciting highlight moment of the video"
MULTI_HIGHLIGHT_PROMPT = (
    "Locate the complete visually distinctive events worth keeping in a short edit. "
    "Keep the start of each action and its outcome. If two events are separated by "
    "unrelated footage, report separate time intervals. Skip routine motion and "
    "static waiting. Report up to three intervals in video seconds, in order. "
    "Reply only in this format: <answer>1.5 to 4.0; 8.0 to 10.0</answer>."
)


def extract_time(text):
    """(start, end) pairs in seconds from the model answer (TimeLens-style parser)."""
    text = text.lower()
    blocks = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.S)
    if blocks:
        text = "; ".join(blocks)
    elif "<answer>" in text:
        text = text[text.rfind("<answer>") + len("<answer>"):]
    number = r"(\d+(?:\.\d+)?)"
    pairs = re.findall(number + r"(?:\s*[-–—]\s*|\s+to\s+)" + number, text)
    return [(float(a), float(b)) for a, b in pairs]


def answer_segments(answer, duration):
    """[(start_sec, end_sec)] from a grounding answer, clipped to [0, duration]."""
    segs = []
    for s, e in extract_time(answer or ""):
        s, e = max(0.0, min(s, e)), min(duration, max(s, e))
        if e > s:
            segs.append((s, e))
    return segs


def clean_caption(text):
    """First sentence of a free-form answer, or '' if it looks like coordinates/garbage."""
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.split(r"(?<=[.!?])\s", " ".join(text.split()))[0].strip().rstrip(".")
    letters = sum(c.isalpha() for c in text)
    if len(text) < 8 or letters < 0.6 * len(text.replace(" ", "")):
        return ""
    return text.replace('"', "'")[:200]


def fit_frame(frame_bgr, max_pixels, factor=32):
    """BGR frame -> RGB array resized to <= max_pixels, sides multiple of `factor`."""
    h, w = frame_bgr.shape[:2]
    s = min(1.0, (max_pixels / float(h * w)) ** 0.5)
    nh = max(factor, int(h * s) // factor * factor)
    nw = max(factor, int(w * s) // factor * factor)
    return cv2.cvtColor(cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


class VideoLLM:
    def __init__(self, model_path, quant="4bit", device="cuda:0", offload=0.0):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        kw = {"dtype": torch.bfloat16, "device_map": {"": device}}
        if offload > 0:  # 9B: embed+lm_head are untied bf16 (4.1 GB) -> park on CPU,
            kw["device_map"] = {"model.language_model.embed_tokens": "cpu",  # layers stay on GPU
                                "lm_head": "cpu", "": device}
        skip = ["visual", "lm_head", "embed_tokens"]
        if quant == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=skip,
                llm_int8_enable_fp32_cpu_offload=offload > 0)
        elif quant == "8bit":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **kw).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.torch = torch

    def sample_frames(self, path, n_frames, fps, sample_fps=2.0, max_frames=128, max_pixels=256 * 448,
                      token_budget=3000, offset=0.0):
        """Uniformly sampled RGB frames (even count, as the model merges pairs) + their ids.
        Frame size shrinks so the video costs <= token_budget tokens (one token per 32x32
        patch per frame pair); ~3000 tokens peaks near 6 GB with 4-bit weights.
        offset shifts the grid by that fraction of one sampling step (0 = the usual grid)."""
        k = int(min(max_frames, max(4, round(n_frames / max(fps, 1e-6) * sample_fps))))
        k -= k % 2
        max_pixels = min(max_pixels, int(token_budget * 2 * 1024 / k))
        pos = np.linspace(0, n_frames - 1, k)
        if offset:
            pos = np.minimum(pos + offset * (n_frames - 1) / max(1, k - 1), n_frames - 1)
        ids = np.unique(pos.round().astype(int)).tolist()
        frames, got = [], []
        for fid, fr in iter_frames(path, ids):
            frames.append(fit_frame(fr, max_pixels))
            got.append(fid)
        if len(frames) % 2:
            frames, got = frames[:-1], got[:-1]
        return frames, got

    def ask_video(self, frames, frame_ids, fps, n_frames, prompt, max_new_tokens=96):
        from transformers.video_utils import VideoMetadata
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        h, w = frames[0].shape[:2]
        meta = VideoMetadata(total_num_frames=n_frames, fps=fps, width=w, height=h,
                             duration=n_frames / fps, frames_indices=list(frame_ids))
        inputs = self.processor(text=[text], videos=[np.stack(frames)], video_metadata=[meta],
                                do_sample_frames=False, do_resize=False, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def highlight_segments(self, frames, frame_ids, fps, n_frames, mode="caption"):
        """[(start_sec, end_sec)], raw answers. mode: 'generic' or 'caption'
        (describe the key moment first, then ground that description)."""
        raw = {}
        query = GENERIC_HIGHLIGHT_QUERY
        if mode == "caption":
            cap = self.ask_video(frames, frame_ids, fps, n_frames, CAPTION_PROMPT, max_new_tokens=48)
            raw["caption"] = cap
            query = clean_caption(cap) or GENERIC_HIGHLIGHT_QUERY
        prompt = MULTI_HIGHLIGHT_PROMPT if mode == "highlights" else TEMPORAL_GROUNDING_PROMPT.format(query)
        ans = self.ask_video(frames, frame_ids, fps, n_frames, prompt)
        raw["grounding"] = ans
        return answer_segments(ans, n_frames / fps), raw
