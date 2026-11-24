"""Normalize browser media and render exact frame ranges with matching audio."""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import subprocess
import tempfile

import av
import cv2
import imageio_ffmpeg
import numpy as np

FPS = 30
MAX_SECONDS = 600


class MediaError(ValueError):
    pass


@dataclass(frozen=True)
class Media:
    width: int
    height: int
    fps: int
    frame_count: int
    duration: float
    has_audio: bool


def ffmpeg() -> list[str]:
    return [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]


def normalize(source: Path, output: Path) -> Media:
    # Restrict the input to ordinary media containers and local I/O. MP4
    # external data references remain disabled (FFmpeg's mov default).
    try:
        subprocess.run(ffmpeg() + [
            "-nostdin", "-protocol_whitelist", "file,pipe",
            "-format_whitelist", "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi", "-i", str(source),
            "-map", "0:V:0", "-map", "0:a:0?", "-map_metadata", "-1", "-map_chapters", "-1",
            "-vf", "fps=30,scale=w='min(1280,iw)':h='min(1280,ih)':"
                   "force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1",
            "-t", str(MAX_SECONDS + 1), "-c:v", "libx264", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-af", "aresample=async=1:first_pts=0",
            "-movflags", "+faststart", str(output),
        ], check=True, capture_output=True, timeout=900)
        with av.open(str(output)) as container:
            stream = container.streams.video[0]
            frames = stream.frames
            if not frames or min(stream.width, stream.height) < 32:
                raise MediaError("Select a video with readable frames and at least 32 pixels per side.")
            if frames > MAX_SECONDS * FPS:
                raise MediaError(f"Select a video no longer than {MAX_SECONDS // 60} minutes.")
            return Media(stream.width, stream.height, FPS, frames, frames / FPS,
                         bool(container.streams.audio))
    except (subprocess.SubprocessError, av.FFmpegError, IndexError) as error:
        raise MediaError("This file could not be decoded. Try an MP4, MOV, WebM, MKV or AVI video.") from error


def render(source: Path, output: Path, media: Media, ratio: str, segments, boxes) -> None:
    rw, rh = map(int, ratio.split(":"))
    unit = min(media.width // rw, media.height // rh) // 2 * 2
    width, height = unit * rw, unit * rh
    args = ffmpeg() + [
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}",
        "-framerate", str(media.fps), "-i", "pipe:0",
    ]
    if media.has_audio:
        trims = []
        for index, (start, end) in enumerate(segments):
            a, b = (round(Fraction(frame * 48000, media.fps)) for frame in (start, end))
            trims.append(f"[1:a]aresample=48000,apad,atrim=start_sample={a}:end_sample={b},"
                         f"asetpts=PTS-STARTPTS[a{index}]")
        inputs = "".join(f"[a{i}]" for i in range(len(segments)))
        trims.append(f"{inputs}concat=n={len(segments)}:v=0:a=1[audio]")
        args += ["-i", str(source), "-filter_complex", ";".join(trims), "-map", "0:v",
                 "-map", "[audio]", "-c:a", "aac"]
    else:
        args += ["-map", "0:v"]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(output)]

    capture = cv2.VideoCapture(str(source))
    try:
        with tempfile.TemporaryFile() as log, subprocess.Popen(args, stdin=subprocess.PIPE, stderr=log) as encoder:
            try:
                frame_index = 0
                for start, end in segments:
                    while frame_index < start:
                        if not capture.grab():
                            raise MediaError("The source video ended before the requested segment.")
                        frame_index += 1
                    while frame_index < end:
                        ok, frame = capture.read()
                        if not ok:
                            raise MediaError("A selected frame could not be decoded.")
                        x, y, w = boxes[frame_index]
                        h = w * rh / rw
                        left = max(0, min(media.width - width, round(x + (w - width) / 2)))
                        top = max(0, min(media.height - height, round(y + (h - height) / 2)))
                        crop = np.ascontiguousarray(frame[top:top + height, left:left + width])
                        encoder.stdin.write(crop.tobytes())
                        frame_index += 1
                encoder.stdin.close()
                if encoder.wait(timeout=120) != 0:
                    log.seek(0)
                    raise MediaError("MP4 encoding failed: " + log.read(1024).decode(errors="replace"))
            finally:
                if encoder.poll() is None:
                    encoder.kill()
    finally:
        capture.release()
