"""Deterministic synthetic images; no webcam or network required."""

import cv2
import numpy as np

from webcam_vision.pose import tag_points


MATRIX = np.array([[800., 0, 640], [0, 810., 360], [0, 0, 1.]])
SIZE = (1280, 720)


def rotation(tilt=(0.15, 0.3, 0.02)):
    return cv2.Rodrigues(np.asarray(tilt, np.float64))[0] @ np.diag([1., -1., -1.])


def render_tag(tag_id=0, translation=(0.03, -0.02, 0.8), tilt=(0.15, 0.3, 0.02), size=0.15):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 400)
    rvec = cv2.Rodrigues(rotation(tilt))[0]
    corners = cv2.projectPoints(tag_points(size), rvec, np.array(translation, np.float64), MATRIX, np.zeros(5))[0].reshape(4, 2)
    source = np.array([[0, 0], [399, 0], [399, 399], [0, 399]], np.float32)
    homography = cv2.getPerspectiveTransform(source, corners.astype(np.float32))
    gray = cv2.warpPerspective(marker, homography, SIZE, flags=cv2.INTER_NEAREST, borderValue=255)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def calibration_data():
    return {"schema_version": 1, "image_size": list(SIZE), "camera_matrix": MATRIX.tolist(),
            "distortion_coefficients": [0.] * 5, "note": "SYNTHETIC TEST ONLY; not a real webcam calibration."}


def render_board(index, columns=9, rows=6, square_size=0.025):
    square_px = 80
    pixels = np.full(((rows + 1) * square_px, (columns + 1) * square_px), 255, np.uint8)
    for row in range(rows + 1):
        for column in range(columns + 1):
            if (row + column) % 2 == 0:
                pixels[row * square_px:(row + 1) * square_px, column * square_px:(column + 1) * square_px] = 0
    rng = np.random.default_rng(index + 42)
    rvec = rng.uniform(-0.35, 0.35, 3)
    translation = np.array([-0.1 + rng.uniform(-0.09, 0.09), -0.06 + rng.uniform(-0.04, 0.04), rng.uniform(0.6, 0.95)])
    outer = np.array([[-1, -1, 0], [columns, -1, 0], [columns, rows, 0], [-1, rows, 0]], np.float64) * square_size
    corners = cv2.projectPoints(outer, rvec, translation, MATRIX, np.zeros(5))[0].reshape(4, 2)
    height, width = pixels.shape
    source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    homography = cv2.getPerspectiveTransform(source, corners.astype(np.float32))
    return cv2.warpPerspective(pixels, homography, SIZE, flags=cv2.INTER_LINEAR, borderValue=255)
