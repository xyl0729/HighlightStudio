import unittest

from studio.final_engine import project_result
from studio.media import Media


class FinalAdapterTest(unittest.TestCase):
    def test_nonmatching_fps_and_display_scale_preserve_delivered_ranges(self):
        native = {"W": 720, "H": 1034, "n": 100, "fps": 25}
        preview = Media(360, 518, 30, 120, 4, True)
        rows = [{"frame": i, "bboxes": [0, 192, 720]}
                for i in list(range(5, 20)) + list(range(50, 75))]
        segments, boxes = project_result(rows, native, preview, (16, 9))
        self.assertEqual(segments, [(6, 24), (60, 90)])
        self.assertEqual(len(boxes), 120)
        self.assertEqual(boxes[6], [0, 96, 360])
        self.assertEqual(boxes[60], boxes[6])

    def test_native_preview_preserves_selected_frame_boxes_exactly(self):
        native = {"W": 640, "H": 360, "n": 60, "fps": 30}
        preview = Media(640, 360, 30, 60, 2, False)
        source = [{"frame": 3, "bboxes": [50, 0, 202]},
                  {"frame": 4, "bboxes": [55, 0, 202]},
                  {"frame": 9, "bboxes": [70, 0, 202]}]
        segments, boxes = project_result(source, native, preview, (9, 16))
        self.assertEqual(segments, [(3, 5), (9, 10)])
        self.assertEqual([boxes[i] for i in (3, 4, 9)], [row["bboxes"] for row in source])
