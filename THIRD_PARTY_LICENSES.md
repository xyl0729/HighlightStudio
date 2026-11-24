# Third-Party Models & Licenses

Highlight Studio relies on several external open-source models and libraries that are **not bundled** in this repository. Users downloading and running these components locally are responsible for complying with each component's respective license terms.

| Component | Source | License | Notes |
|---|---|---|---|
| **Video-ORA-4B** | https://huggingface.co/OraRL/Video-ORA-4B | Apache-2.0 | Used for 4-bit quantized temporal highlight localization on CUDA. |
| **CLIP ViT-B/32** | https://huggingface.co/openai/clip-vit-base-patch32 | MIT License | Visual semantic feature extraction (see upstream model card for details). |
| **Whisper small** | https://huggingface.co/openai/whisper-small | Apache-2.0 | Audio / speech feature extraction (Hugging Face model card license: Apache-2.0; OpenAI Whisper repository code: MIT). |
| **UNISAL** | https://github.com/rdroste/unisal | Apache-2.0 | Visual saliency estimation for dynamic reframing (Apache-2.0 at repository root; upstream repository does not include a separate weight license file). |
| **Ultralytics YOLO11s** | https://github.com/ultralytics/ultralytics | AGPL-3.0 / Commercial | Subject detection. Note that Ultralytics packages are licensed under AGPL-3.0 unless a commercial license is obtained from Ultralytics. |
| **FFmpeg** (via `imageio-ffmpeg`) | https://ffmpeg.org/ | LGPL / GPL (depends on build) | Media normalization, frame-accurate cropping, and audio muxing. |
| **FastAPI / React / Vite / Lucide** | PyPI / npm | MIT / ISC | Web service and browser NLE workspace dependencies. |
