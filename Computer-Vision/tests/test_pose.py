import unittest

import cv2
import numpy as np

from webcam_vision.pose import Candidate, PoseEstimator, TagTracker, invert_transform, rotation_distance, transform
from helpers import MATRIX, SIZE, render_tag, rotation


class PoseTests(unittest.TestCase):
    def test_inverse_round_trip(self):
        pose = transform(rotation(), [0.2, -0.1, 1.2])
        np.testing.assert_allclose(invert_transform(pose) @ pose, np.eye(4), atol=1e-12)
        point = np.array([0.4, 0.3, 0.1, 1.])
        np.testing.assert_allclose(invert_transform(pose) @ (pose @ point), point, atol=1e-12)

    def test_synthetic_pose_with_distortion(self):
        distortion = np.array([-0.12, 0.03, 0.001, -0.002, 0.005])
        estimator = PoseEstimator(MATRIX, distortion, 0.15)
        for translation, tilt in [((0.03, -0.02, 0.5), (0.15, 0.3, 0.02)),
                                  ((-0.18, 0.07, 1.), (-0.2, 0.4, 0.1)),
                                  ((0.25, -0.1, 2.), (0.35, -0.4, -0.2))]:
            with self.subTest(translation=translation):
                r = rotation(tilt)
                corners = cv2.projectPoints(estimator.points, cv2.Rodrigues(r)[0], np.array(translation), MATRIX, distortion)[0]
                # At long range a correctly conservative selector may report ambiguity;
                # verify the exact solution among raw hypotheses, not a forced selection.
                candidates = estimator.candidates(corners)
                self.assertGreater(len(candidates), 0)
                best = candidates[0]
                np.testing.assert_allclose(best.camera_from_tag, transform(r, translation), atol=1e-7)
                np.testing.assert_allclose(best.tag_from_camera[:3, 3], -r.T @ translation, atol=1e-7)
                self.assertLess(best.rms_px, 1e-7)

    def test_rendered_detection_and_decoded_orientation(self):
        for angle in [0.02, 1.1, -1.5]:
            tracker = TagTracker(MATRIX, np.zeros(5), .15)
            record, _, _, _ = tracker.process(render_tag(tilt=(.15, .3, angle)), 0)
            self.assertTrue(record["valid"], record["status"])
            expected = invert_transform(transform(rotation((.15, .3, angle)), [.03, -.02, .8]))
            np.testing.assert_allclose(record["camera_position_m"], expected[:3, 3], atol=.025)
            self.assertLess(rotation_distance(np.asarray(record["T_tag_from_camera"])[:3, :3], expected[:3, :3]), .06)

    def test_loss_and_wrong_id_clear_pose(self):
        tracker = TagTracker(MATRIX, np.zeros(5), .15)
        self.assertTrue(tracker.process(render_tag(), 0)[0]["valid"])
        for frame in [np.full((SIZE[1], SIZE[0], 3), 255, np.uint8), render_tag(tag_id=1)]:
            record = tracker.process(frame, .1)[0]
            self.assertFalse(record["valid"])
            self.assertIsNone(record["T_tag_from_camera"])
            self.assertIsNone(record["camera_position_m"])
            self.assertIsNone(tracker.estimator.previous)

    def test_duplicate_id_rejected(self):
        image = np.minimum(render_tag(translation=(-.22, 0, 1)), render_tag(translation=(.22, 0, 1)))
        record = TagTracker(MATRIX, np.zeros(5), .15).process(image, 0)[0]
        self.assertEqual(record["status"], "duplicate_target_id")
        self.assertFalse(record["valid"])

    def test_ambiguity_needs_recent_distinguishing_history(self):
        estimator = PoseEstimator(MATRIX, np.zeros(5), .15)
        a = Candidate(transform(rotation(), [.03, 0, .8]), np.zeros((3, 1)), np.zeros((3, 1)), .1)
        b = Candidate(transform(rotation((-.15, -.3, .02)), [.03, 0, .8]), np.zeros((3, 1)), np.zeros((3, 1)), .12)
        selected, status = estimator.select([a, b], 0)
        self.assertIsNone(selected)
        self.assertEqual(status, "ambiguous_pose")
        estimator.previous, estimator.previous_time = a, 0
        self.assertIs(estimator.select([a, b], .1)[0], a)
        self.assertIsNone(estimator.select([a, b], 1)[0])

    def test_invalid_or_tiny_corners(self):
        estimator = PoseEstimator(MATRIX, np.zeros(5), .15)
        for corners in [np.zeros((4, 2)), np.full((4, 2), np.nan)]:
            self.assertFalse(estimator.estimate(corners, 0)[0])


if __name__ == "__main__":
    unittest.main()
