import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from webcam_vision.calibration import load_calibration
from webcam_vision.cli import build_parser, main
from webcam_vision.common import check_size, write_json
from webcam_vision.tracking import annotate, read_timestamps, run
from webcam_vision.pose import TagTracker
from helpers import MATRIX, SIZE, calibration_data, render_board, render_tag


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calibration = self.root / "calibration.json"
        write_json(self.calibration, calibration_data())

    def call(self, arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main([str(x) for x in arguments])

    def test_calibration_validation_and_resolution(self):
        _, _, _, size = load_calibration(self.calibration)
        self.assertEqual(size, SIZE)
        with self.assertRaisesRegex(ValueError, "resolution"):
            check_size(np.zeros((10, 20, 3)), size)
        data = calibration_data()
        data["camera_matrix"][0][0] = -1
        invalid = self.root / "invalid.json"
        write_json(invalid, data)
        with self.assertRaises(ValueError):
            load_calibration(invalid)

    def test_generated_targets_are_detectable(self):
        output = self.root / "targets"
        self.assertEqual(self.call(["targets", "--output", output]), 0)
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
        ids = detector.detectMarkers(cv2.imread(str(output / "tag36h11_0.png")))[1]
        self.assertEqual(ids.reshape(-1).tolist(), [0])
        from webcam_vision.calibration import find_board
        self.assertIsNotNone(find_board(cv2.imread(str(output / "checkerboard.png"), 0), 9, 6))
        self.assertEqual(self.call(["targets", "--output", output]), 1)

    def test_calibration_fit_with_independent_views(self):
        train, validation = self.root / "train", self.root / "validation"
        train.mkdir()
        validation.mkdir()
        for index in range(20):
            folder = train if index < 15 else validation
            self.assertTrue(cv2.imwrite(str(folder / f"{index}.png"), render_board(index)))
        output = self.root / "fitted.json"
        previews = self.root / "previews"
        self.assertEqual(self.call(["calibrate", "--images", train, "--validation", validation,
                                    "--square-size", ".025", "--output", output, "--previews", previews]), 0)
        result, k, _, _ = load_calibration(output)
        self.assertLess(result["held_out_rms_px"], 1)
        np.testing.assert_allclose([k[0, 0], k[1, 1]], [MATRIX[0, 0], MATRIX[1, 1]], rtol=.03)
        self.assertEqual(len(list(previews.glob("*.png"))), 5)
        self.assertEqual(self.call(["calibrate", "--images", train, "--validation", train,
                                    "--square-size", ".025", "--output", self.root / "bad.json"]), 1)

    def make_video(self):
        path = self.root / "source.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, SIZE)
        self.assertTrue(writer.isOpened())
        for index in range(8):
            image = render_tag() if index < 4 else render_tag(tag_id=1)
            writer.write(image)
        writer.release()
        return path

    def test_replay_recording_and_analysis(self):
        source = self.make_video()
        output = self.root / "run"
        self.assertEqual(self.call(["replay", "--video", source, "--calibration", self.calibration,
                                    "--tag-size", ".15", "--output", output, "--headless", "--record"]), 0)
        rows = [json.loads(line) for line in (output / "poses.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["valid"] for row in rows[:4]))
        self.assertTrue(all(row["camera_position_m"] is None for row in rows[4:]))
        self.assertEqual(read_timestamps(output / "poses.jsonl"), [i / 15 for i in range(8)])
        report = self.root / "report.json"
        self.assertEqual(self.call(["analyze", "--log", output / "poses.jsonl", "--stationary",
                                    "--distance", str(np.linalg.norm([.03, -.02, .8])), "--output", report]), 0)
        analysis = json.loads(report.read_text())
        self.assertLess(analysis["median_absolute_range_error_percent"], 5)
        self.assertEqual(analysis["valid_fraction"], .5)
        self.assertEqual(self.call(["replay", "--video", output / "camera.avi", "--timestamps", output / "poses.jsonl",
                                    "--calibration", self.calibration, "--tag-size", ".15", "--headless",
                                    "--output", self.root / "replayed"]), 0)

    def test_bad_timestamps_fail(self):
        path = self.root / "time.jsonl"
        path.write_text('\n'.join(json.dumps({"frame_index": i, "timestamp_s": 0}) for i in range(2)))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            read_timestamps(path)

    def test_camera_disconnect_logs_invalid_terminal_state_and_releases(self):
        class Camera:
            def __init__(self):
                self.reads = 0
                self.released = False

            def get(self, prop):
                return 30

            def read(self):
                self.reads += 1
                return (True, render_tag()) if self.reads == 1 else (False, None)

            def release(self):
                self.released = True

        camera = Camera()
        output = self.root / "disconnect"
        args = build_parser().parse_args(["track", "--calibration", str(self.calibration), "--tag-size", ".15",
                                          "--output", str(output), "--headless"])
        with patch("webcam_vision.tracking.open_camera", return_value=(camera, [])), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "stopped delivering"):
                run(args)
        self.assertTrue(camera.released)
        rows = [json.loads(line) for line in (output / "poses.jsonl").read_text().splitlines()]
        self.assertTrue(rows[0]["valid"])
        self.assertFalse(rows[-1]["valid"])
        self.assertIsNone(rows[-1]["T_tag_from_camera"])

    def test_overlay_renders_valid_and_missing_states(self):
        tracker = TagTracker(MATRIX, np.zeros(5), .15)
        for image in [render_tag(), np.full((SIZE[1], SIZE[0], 3), 255, np.uint8)]:
            row, corners, ids, selected = tracker.process(image, 0)
            row["processing_ms"] = 2.
            overlay = annotate(image, row, corners, ids, selected, tracker, 30, 4)
            self.assertEqual(overlay.shape, image.shape)
            self.assertFalse(np.array_equal(image, overlay))


if __name__ == "__main__":
    unittest.main()
