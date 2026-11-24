import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import av
import imageio_ffmpeg
from studio.app import create_app
from fastapi.testclient import TestClient


class BackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.sample = cls.root / "generated.mp4"
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", "3", "-c:v", "libx264", "-c:a", "aac", str(cls.sample),
        ], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def wait(self, client, uri, status):
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            data = client.get(uri).json()
            if data["status"] == status:
                return data
            self.assertNotEqual(data["status"], "failed", data)
            time.sleep(0.1)
        self.fail(f"Timed out waiting for {status}: {data}")

    def test_import_edit_and_export_both_ratios_with_audio(self):
        with TestClient(create_app(self.root / "jobs", dev_preview=True)) as client:
            for ratio in ("9:16", "16:9"):
                with self.subTest(ratio=ratio):
                    with self.sample.open("rb") as f:
                        response = client.post("/api/jobs", files={"file": ("self-made.mp4", f)},
                                               data={"aspect_ratio": ratio, "mode": "quick"})
                    self.assertEqual(response.status_code, 202, response.text)
                    uri = "/api/jobs/" + response.json()["id"]
                    job = self.wait(client, uri, "ready")
                    self.assertEqual(job["analysis_engine"], "motion")
                    self.assertEqual(len(job["crop_boxes"]), job["media"]["frame_count"])
                    self.assertEqual(client.get(job["source_url"], headers={"Range": "bytes=0-99"}).status_code, 206)
                    invalid = {"segments": [{"start_frame": 0, "end_frame": job["media"]["frame_count"] + 1}]}
                    self.assertEqual(client.put(uri + "/segments", json=invalid).status_code, 422)
                    segments = [{"start_frame": 0, "end_frame": 12}, {"start_frame": 30, "end_frame": 48}]
                    revised = client.put(uri + "/segments", json={"segments": segments})
                    self.assertEqual(revised.status_code, 200, revised.text)
                    self.assertEqual(revised.json()["segments"], segments)
                    self.assertEqual(client.post(uri + "/export").status_code, 202)
                    exported = self.wait(client, uri, "exported")
                    result = client.get(exported["download_url"])
                    self.assertEqual(result.status_code, 200)
                    output = self.root / "render.mp4"
                    output.write_bytes(result.content)
                    with av.open(str(output)) as video:
                        stream = video.streams.video[0]
                        w, h = map(int, ratio.split(":"))
                        self.assertEqual(stream.width * h, stream.height * w)
                        self.assertEqual(len(list(video.decode(video=0))), 30)
                    with av.open(str(output)) as video:
                        self.assertTrue(video.streams.audio)
                        self.assertAlmostEqual(float(video.streams.audio[0].duration * video.streams.audio[0].time_base),
                                               1.0, delta=0.08)
                    self.assertEqual(client.put(uri + "/segments", json={"segments": segments}).status_code, 200)
                    self.assertEqual(client.get(exported["download_url"]).status_code, 409)
                    self.assertEqual(client.delete(uri).status_code, 204)
                    self.assertEqual(client.get(uri).status_code, 404)

    def test_saved_edits_and_export_survive_backend_restart(self):
        work = self.root / "restart"
        with TestClient(create_app(work, dev_preview=True)) as client:
            with self.sample.open("rb") as f:
                created = client.post("/api/jobs", files={"file": ("demo.mp4", f)},
                                      data={"mode": "quick", "aspect_ratio": "16:9"})
            self.assertEqual(created.status_code, 202)
            job_id = created.json()["id"]
            uri = "/api/jobs/" + job_id
            original = self.wait(client, uri, "ready")
            edits = [{"start_frame": 3, "end_frame": 24}]
            self.assertEqual(client.put(uri + "/segments", json={"segments": edits}).status_code, 200)
            self.assertEqual(client.post(uri + "/export").status_code, 202)
            exported = self.wait(client, uri, "exported")
            download = client.get(exported["download_url"]).content

        with TestClient(create_app(work, dev_preview=True)) as client:
            history = client.get("/api/jobs")
            self.assertEqual(history.status_code, 200, history.text)
            self.assertEqual([(job["id"], job["name"], job["status"]) for job in history.json()],
                             [(job_id, "demo.mp4", "exported")])
            self.assertEqual(history.json()[0]["crop_boxes"], [])
            restored = client.get(uri).json()
            self.assertEqual(restored["segments"], edits)
            self.assertEqual(restored["recommended_segments"], original["segments"])
            self.assertEqual(restored["crop_boxes"], original["crop_boxes"])
            self.assertEqual(client.get(restored["download_url"]).content, download)
            self.assertEqual(client.delete(uri).status_code, 204)

        with TestClient(create_app(work, dev_preview=True)) as client:
            self.assertEqual(client.get("/api/jobs").json(), [])

    def test_invalid_media_and_upload_limit(self):
        with TestClient(create_app(self.root / "invalid", max_upload_bytes=1024, dev_preview=True)) as client:
            large = client.post("/api/jobs", files={"file": ("large.mp4", b"x" * 1025)},
                                data={"mode": "quick"})
            self.assertEqual(large.status_code, 413)
            bad = client.post("/api/jobs", files={"file": ("invalid.mp4", b"this is not a video")},
                              data={"mode": "quick"})
            self.assertEqual(bad.status_code, 202)
            result = client.get("/api/jobs/" + bad.json()["id"]).json()
            deadline = time.monotonic() + 10
            while result["status"] != "failed" and time.monotonic() < deadline:
                time.sleep(0.1)
                result = client.get("/api/jobs/" + bad.json()["id"]).json()
            self.assertEqual(result["status"], "failed", result)
            self.assertEqual(result["error"]["code"], "invalid_video")
