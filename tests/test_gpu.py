import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import av
import imageio_ffmpeg
from fastapi.testclient import TestClient

from studio.app import create_app


@unittest.skipUnless(os.getenv("HIGHLIGHT_MODEL_DIR"), "Set HIGHLIGHT_MODEL_DIR to run the CUDA model smoke test")
class GPUAcceptance(unittest.TestCase):
    def test_real_model_upload_and_export(self):
        with tempfile.TemporaryDirectory() as temp, TestClient(create_app(Path(temp) / "jobs", dev_preview=True)) as client:
            sample = Path(temp) / "synthetic.mp4"
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error",
                "-f", "lavfi", "-i", "color=navy:size=384x216:rate=30",
                "-f", "lavfi", "-i", "color=yellow:size=64x64:rate=30",
                "-filter_complex", "[0:v][1:v]overlay=x='if(lt(t,2),20,min(290,20+(t-2)*135))':y=76",
                "-t", "6", "-c:v", "libx264", str(sample),
            ], check=True)
            with sample.open("rb") as media:
                response = client.post("/api/jobs", files={"file": ("synthetic.mp4", media)})
            self.assertEqual(response.status_code, 202, response.text)
            uri = "/api/jobs/" + response.json()["id"]
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                job = client.get(uri).json()
                if job["status"] in {"ready", "failed"}:
                    break
                time.sleep(1)
            self.assertEqual(job["status"], "ready", job.get("error"))
            self.assertEqual(job["analysis_engine"], "video-ora-4b", job["warnings"])
            self.assertTrue(job["segments"])
            print("GPU segments:", job["segments"], flush=True)
            self.assertEqual(client.post(uri + "/export").status_code, 202)
            for _ in range(60):
                job = client.get(uri).json()
                if job["status"] != "exporting":
                    break
                time.sleep(0.5)
            self.assertEqual(job["status"], "exported", job["error"])
            output = Path(temp) / "out.mp4"
            output.write_bytes(client.get(job["download_url"]).content)
            with av.open(str(output)) as result:
                video = result.streams.video[0]
                self.assertEqual(video.width * 16, video.height * 9)
                self.assertFalse(result.streams.audio)
