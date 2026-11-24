import subprocess
import tempfile
import unittest
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from studio.analysis import Analyzer
from studio.media import Media
from hlcut.temporal import split_motion_gap
from hlcut.vlm import answer_segments


class ProposedHighlights:
    def sample_frames(self, *args, **kwargs):
        return [None] * 8, list(range(8))

    def highlight_segments(self, frames, ids, fps, total, mode):
        if mode == "highlights":
            return [(1.0, 2.5), (5.0, 6.5)], {}
        return [(1.0, 2.5)], {}


class SubjectDetector:
    def __call__(self, frames):
        return [np.array([[20, 30, 85, 135, 0.90, 16]]) for _ in frames]


class AnalysisTest(unittest.TestCase):
    def test_ai_preserves_two_events_and_crops_to_off_center_subject(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.mp4"
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error",
                "-f", "lavfi", "-i", "color=gray:size=320x180:rate=30",
                "-t", "8", "-c:v", "libx264", str(path),
            ], check=True)
            analyzer = Analyzer(None)
            analyzer.detector = SubjectDetector()
            analyzer.vlm = ProposedHighlights()
            spans, boxes, engine, warnings, tracking = analyzer.analyze(
                path, Media(320, 180, 30, 240, 8.0, False), "9:16", "ai",
            )
            self.assertEqual(len(spans), 2)
            self.assertLess(spans[0][1], spans[1][0])
            self.assertEqual(engine, "video-ora-4b")
            self.assertEqual(tracking, "subject")
            self.assertEqual(warnings, [])
            self.assertTrue(all(x <= 20 and x + w >= 85 for x, _, w in boxes))

    def test_motion_follows_small_target_instead_of_centering(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.mp4"
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error",
                "-f", "lavfi", "-i", "color=black:size=640x360:rate=30",
                "-f", "lavfi", "-i", "color=white:size=12x12:rate=30",
                "-filter_complex", "[0:v][1:v]overlay=x='30+t*140':y=170",
                "-t", "4", "-c:v", "libx264", str(path),
            ], check=True)
            _, boxes, _, _, tracking = Analyzer(None).analyze(
                path, Media(640, 360, 30, 120, 4.0, False), "9:16", "quick",
            )
            self.assertEqual(tracking, "motion")
            self.assertGreater(boxes[-1][0] - boxes[0][0], 250)

    def test_mixed_timestamp_separators_preserve_both_events(self):
        self.assertEqual(answer_segments("<answer> 1 to 2; 5-6 </answer>", 8), [(1, 2), (5, 6)])

    def test_long_window_splits_only_at_a_sustained_gap(self):
        scores = np.zeros(36)
        scores[5:15] = 1
        scores[22:32] = 0.95
        ids = np.arange(36) * 8
        cuts = split_motion_gap([(0, 288)], ids, scores, 30, 288)
        self.assertEqual(len(cuts), 2)
        self.assertLess(cuts[0][1], 18 * 8)
        self.assertGreater(cuts[1][0], 18 * 8)
        scores[15:22] = 0.3
        self.assertEqual(split_motion_gap([(0, 288)], ids, scores, 30, 288), [(0, 288)])
        original = [(0, 115), (164, 288)]
        self.assertEqual(split_motion_gap(original, ids, scores, 30, 288), original)
