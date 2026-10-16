"""Shared I/O and camera helpers. All distances are meters."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing a calibration or run.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("must be finite and greater than zero")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise ValueError("must be greater than zero")
    return number


def require_gui():
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:") and "NONE" in line:
            raise ValueError(
                "This OpenCV build has no GUI. Install the project in a fresh virtual "
                "environment (README), or use --headless for tracking/replay."
            )


def open_camera(index, width, height, fps=30, focus=None, exposure=None):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open camera {index}; check device permissions and other camera apps.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    warnings = []
    if focus is not None:
        for prop, value, label in [
            (cv2.CAP_PROP_AUTOFOCUS, 0, "disable autofocus"),
            (cv2.CAP_PROP_FOCUS, focus, "set focus"),
        ]:
            if not cap.set(prop, value):
                warnings.append(f"Camera did not acknowledge {label}; check its control panel.")
    if exposure is not None:
        # Auto-exposure modes are backend-specific; don't guess a magic mode value.
        if not cap.set(cv2.CAP_PROP_EXPOSURE, exposure):
            warnings.append("Camera did not acknowledge exposure; set manual mode in its control panel.")
    return cap, warnings


def camera_settings(cap):
    properties = {
        "width": cv2.CAP_PROP_FRAME_WIDTH,
        "height": cv2.CAP_PROP_FRAME_HEIGHT,
        "fps": cv2.CAP_PROP_FPS,
        "focus": cv2.CAP_PROP_FOCUS,
        "autofocus": cv2.CAP_PROP_AUTOFOCUS,
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        "zoom": cv2.CAP_PROP_ZOOM,
    }
    values = {name: float(cap.get(prop)) for name, prop in properties.items()}
    return {name: value if math.isfinite(value) else None for name, value in values.items()}


def check_size(frame, size):
    actual = (frame.shape[1], frame.shape[0])
    if actual != tuple(size):
        raise ValueError(
            f"Image resolution {actual} does not match required {tuple(size)}. "
            "Select the calibrated mode or recalibrate; images are never silently resized."
        )


def put_lines(image, lines, color=(240, 240, 240)):
    for index, line in enumerate(lines):
        origin = (12, 25 + 24 * index)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
