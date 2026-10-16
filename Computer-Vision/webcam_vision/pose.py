"""AprilTag pose geometry and explicit planar-ambiguity handling.

T_a_from_b maps points in b to a. Camera axes: right/down/forward.
Tag axes: decoded right/up/out of the printed face. Units: meters/radians.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np


def tag_points(size):
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Tag size must be finite and positive, in meters.")
    half = size / 2
    # ArucoDetector reports decoded TL, TR, BR, BL, exactly IPPE_SQUARE's order.
    return np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float64)


def transform(rotation, translation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation).reshape(3)
    return result


def invert_transform(matrix):
    rotation = matrix[:3, :3]
    return transform(rotation.T, -rotation.T @ matrix[:3, 3])


def rotation_distance(a, b):
    """Geodesic rotation difference in radians; no Euler-angle wrapping issues."""
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))


def rpy_degrees(rotation):
    """Display only: R = Rz(yaw) Ry(pitch) Rx(roll), camera expressed in tag frame."""
    horizontal = np.hypot(rotation[0, 0], rotation[1, 0])
    if horizontal > 1e-6:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    pitch = np.arctan2(-rotation[2, 0], horizontal)
    return np.degrees([roll, pitch, yaw]).tolist()


@dataclass
class PosePolicy:
    max_reprojection_px: float = 2.5
    ambiguity_gap_px: float = 0.25
    min_side_px: float = 12.0
    history_max_age_s: float = 0.25
    equivalent_rotation_deg: float = 2.0
    equivalent_translation_m: float = 0.002
    continuity_margin: float = 0.25
    max_history_translation_m: float = 0.5
    max_history_rotation_deg: float = 45.0


@dataclass
class Candidate:
    camera_from_tag: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    rms_px: float

    @property
    def tag_from_camera(self):
        return invert_transform(self.camera_from_tag)

    def record(self):
        return {"T_camera_from_tag": self.camera_from_tag.tolist(), "reprojection_rms_px": self.rms_px}


class PoseEstimator:
    def __init__(self, matrix, distortion, tag_size, policy=None):
        self.matrix = np.asarray(matrix, np.float64)
        self.distortion = np.asarray(distortion, np.float64)
        self.points = tag_points(tag_size)
        self.tag_size = tag_size
        self.policy = policy or PosePolicy()
        self.previous = None
        self.previous_time = None

    def reset(self):
        self.previous = None
        self.previous_time = None

    def candidates(self, corners):
        corners = np.asarray(corners, np.float64).reshape(4, 2)
        if not np.all(np.isfinite(corners)):
            return []
        try:
            count, rotations, translations, _ = cv2.solvePnPGeneric(
                self.points, corners, self.matrix, self.distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
        except cv2.error:
            return []
        result = []
        if not count:
            return result
        for rvec, tvec in zip(rotations, translations):
            if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
                continue
            rotation = cv2.Rodrigues(rvec)[0]
            camera_points = (rotation @ self.points.T).T + tvec.reshape(3)
            if np.any(camera_points[:, 2] <= 0):
                continue
            pose = transform(rotation, tvec)
            # Printed front face must face the observing camera.
            if invert_transform(pose)[2, 3] <= 0:
                continue
            projected = cv2.projectPoints(self.points, rvec, tvec, self.matrix, self.distortion)[0].reshape(4, 2)
            error = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))
            if np.isfinite(error):
                result.append(Candidate(pose, rvec, tvec, error))
        return sorted(result, key=lambda value: value.rms_px)

    def select(self, candidates, timestamp):
        """Return an accepted raw candidate or a reason; never substitute an old pose."""
        policy = self.policy
        good = [candidate for candidate in candidates if candidate.rms_px <= policy.max_reprojection_px]
        if not good:
            return None, "high_reprojection_error" if candidates else "no_physical_pose"
        good.sort(key=lambda value: value.rms_px)
        best = good[0]
        if len(good) > 1 and good[1].rms_px - best.rms_px <= policy.ambiguity_gap_px:
            a, b = best.tag_from_camera, good[1].tag_from_camera
            equivalent = (
                rotation_distance(a[:3, :3], b[:3, :3]) <= np.radians(policy.equivalent_rotation_deg)
                and np.linalg.norm(a[:3, 3] - b[:3, 3]) <= policy.equivalent_translation_m
            )
            if not equivalent:
                recent = (
                    self.previous is not None and self.previous_time is not None
                    and 0 <= timestamp - self.previous_time <= policy.history_max_age_s
                )
                if not recent:
                    return None, "ambiguous_pose"
                previous = self.previous.tag_from_camera
                scored = []
                for candidate in good[:2]:
                    current = candidate.tag_from_camera
                    translation = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
                    angle = rotation_distance(current[:3, :3], previous[:3, :3])
                    if translation <= policy.max_history_translation_m and angle <= np.radians(policy.max_history_rotation_deg):
                        score = translation / self.tag_size + angle / np.radians(30)
                        scored.append((score, candidate))
                scored.sort(key=lambda item: item[0])
                if not scored or (len(scored) > 1 and scored[1][0] - scored[0][0] < policy.continuity_margin):
                    return None, "ambiguous_pose"
                best = scored[0][1]
        self.previous, self.previous_time = best, timestamp
        return best, "tracking"

    def estimate(self, corners, timestamp):
        corners = np.asarray(corners, np.float64).reshape(4, 2)
        lengths = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
        if not np.all(np.isfinite(corners)) or min(lengths) < self.policy.min_side_px:
            self.reset()
            return None, "tag_too_small_or_invalid", []
        candidates = self.candidates(corners)
        selected, status = self.select(candidates, timestamp)
        if selected is None:
            self.reset()
        return selected, status, candidates


class TagTracker:
    def __init__(self, matrix, distortion, tag_size, target_id=0, policy=None):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        if not 0 <= target_id < self.dictionary.bytesList.shape[0]:
            raise ValueError(f"Target ID must be between 0 and {self.dictionary.bytesList.shape[0] - 1}.")
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, parameters)
        self.estimator = PoseEstimator(matrix, distortion, tag_size, policy)
        self.target_id = target_id

    def process(self, frame, timestamp):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self.detector.detectMarkers(gray)
        detected_ids = [] if ids is None else ids.reshape(-1).tolist()
        record = {
            "timestamp_s": float(timestamp), "tag_id": self.target_id,
            "detected_ids": detected_ids, "valid": False, "status": "target_not_found",
            "T_tag_from_camera": None, "T_camera_from_tag": None,
            "camera_position_m": None, "camera_rpy_deg": None, "distance_m": None,
            "reprojection_rms_px": None, "corners_px": None, "raw_candidates": [],
        }
        matches = [index for index, tag_id in enumerate(detected_ids) if tag_id == self.target_id]
        selected = None
        if len(matches) == 1:
            target_corners = corners[matches[0]].reshape(4, 2)
            selected, status, candidates = self.estimator.estimate(target_corners, timestamp)
            record.update(status=status, corners_px=target_corners.tolist(), raw_candidates=[c.record() for c in candidates])
        else:
            self.estimator.reset()
            if len(matches) > 1:
                record["status"] = "duplicate_target_id"
        if selected is not None:
            camera_pose = selected.tag_from_camera
            record.update(
                valid=True, T_tag_from_camera=camera_pose.tolist(), T_camera_from_tag=selected.camera_from_tag.tolist(),
                camera_position_m=camera_pose[:3, 3].tolist(), camera_rpy_deg=rpy_degrees(camera_pose[:3, :3]),
                distance_m=float(np.linalg.norm(camera_pose[:3, 3])), reprojection_rms_px=selected.rms_px,
            )
        return record, corners, ids, selected

    def metadata(self):
        return {"family": "36h11", "target_id": self.target_id, "tag_size_m": self.estimator.tag_size,
                "pose_policy": asdict(self.estimator.policy)}
