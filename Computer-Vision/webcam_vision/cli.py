"""Command-line entry points; run python -m webcam_vision --help."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from . import calibration, tracking
from .common import positive_float, positive_int, write_json


def generate_targets(args):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if not 0 <= args.tag_id < dictionary.bytesList.shape[0]:
        raise ValueError(f"Tag ID must be between 0 and {dictionary.bytesList.shape[0] - 1}.")
    calibration.board_points(args.columns, args.rows, 1.0)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    marker = cv2.aruco.generateImageMarker(dictionary, args.tag_id, 800)
    marker = cv2.copyMakeBorder(marker, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
    square_px = 100
    board = np.full(((args.rows + 1) * square_px, (args.columns + 1) * square_px), 255, np.uint8)
    for row in range(args.rows + 1):
        for column in range(args.columns + 1):
            if (row + column) % 2 == 0:
                board[row * square_px:(row + 1) * square_px, column * square_px:(column + 1) * square_px] = 0
    board = cv2.copyMakeBorder(board, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
    for name, image in [(f"tag36h11_{args.tag_id}.png", marker), ("checkerboard.png", board)]:
        if not cv2.imwrite(str(folder / name), image):
            raise ValueError(f"Could not write {name}.")
    write_json(folder / "targets.json", {
        "family": "36h11", "tag_id": args.tag_id, "tag_marker_pixels": 800, "white_margin_pixels": 100,
        "checkerboard_internal_corners": [args.columns, args.rows], "checkerboard_square_pixels": square_px,
        "note": "PNG pixel size is NOT a physical size. Measure the printed black marker's outer width and each checkerboard square in meters. Preserve aspect ratio; mount flat.",
    })
    print(f"Targets saved in {folder}. Measure actual printed dimensions; do not include the white tag margin.")


def board_arguments(parser, measured=True):
    parser.add_argument("--columns", type=positive_int, default=9, help="Internal corner columns, not squares (default: 9).")
    parser.add_argument("--rows", type=positive_int, default=6, help="Internal corner rows, not squares (default: 6).")
    if measured:
        parser.add_argument("--square-size", type=positive_float, required=True, help="Measured checkerboard square side in meters.")


def camera_arguments(parser, capture=False):
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument("--width", type=positive_int, default=1280 if capture else None,
                        help="Capture width; tracking defaults to calibration width.")
    parser.add_argument("--height", type=positive_int, default=720 if capture else None,
                        help="Capture height; tracking defaults to calibration height.")
    parser.add_argument("--fps", type=positive_float, default=30, help="Requested camera FPS (default: 30).")
    parser.add_argument("--focus", type=float, help="Request manual focus; device-specific units. Verify in camera controls.")
    parser.add_argument("--exposure", type=float, help="Request exposure in device-specific units; set manual mode externally.")


def tracking_arguments(parser):
    parser.add_argument("--calibration", required=True, help="Calibration JSON produced by calibrate.")
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--tag-size", type=positive_float, required=True, help="Measured outer black marker width in meters, excluding white margin.")
    parser.add_argument("--output", help="New run directory; defaults to runs/<UTC timestamp>.")
    parser.add_argument("--headless", action="store_true", help="Log without a window; replay processes as fast as possible.")
    parser.add_argument("--record", action="store_true", help="Record raw frames to camera.avi alongside poses.jsonl.")
    parser.add_argument("--max-frames", type=positive_int, help="Stop after this many frames.")
    parser.add_argument("--max-reprojection", type=positive_float, default=2.5, help="Reject pose above this RMS in pixels (default: 2.5).")
    parser.add_argument("--ambiguity-gap", type=positive_float, default=0.25, help="Treat candidates within this RMS gap as potentially ambiguous (default: 0.25px).")


def build_parser():
    parser = argparse.ArgumentParser(description="Calibrate a webcam and track its pose relative to a stationary AprilTag.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    targets = subparsers.add_parser("targets", help="Generate printable AprilTag and checkerboard PNGs.")
    board_arguments(targets, measured=False)
    targets.add_argument("--tag-id", type=int, default=0)
    targets.add_argument("--output", default="targets", help="New output directory (default: targets).")
    targets.set_defaults(action=generate_targets)

    capture = subparsers.add_parser("capture", help="Interactively save training and held-out checkerboard images.")
    board_arguments(capture)
    camera_arguments(capture, capture=True)
    capture.add_argument("--output", required=True, help="New dataset directory.")
    capture.set_defaults(action=calibration.capture)

    fit = subparsers.add_parser("calibrate", help="Fit intrinsics and evaluate independent held-out images.")
    board_arguments(fit)
    fit.add_argument("--images", required=True, help="Training image directory.")
    fit.add_argument("--validation", required=True, help="Held-out image directory.")
    fit.add_argument("--output", default="calibration.json", help="New calibration JSON file.")
    fit.add_argument("--previews", help="New directory for original/undistorted validation previews.")
    fit.set_defaults(action=calibration.calibrate)

    live = subparsers.add_parser("track", help="Track a tag using a live webcam.")
    camera_arguments(live)
    tracking_arguments(live)
    live.set_defaults(action=tracking.run)

    replay = subparsers.add_parser("replay", help="Track from recorded video, with optional original timestamps.")
    tracking_arguments(replay)
    replay.add_argument("--video", required=True)
    replay.add_argument("--timestamps", help="Original poses.jsonl; restores capture timing, including dropped-frame intervals.")
    replay.set_defaults(action=tracking.run, width=None, height=None, fps=30)

    analyze = subparsers.add_parser("analyze", help="Summarize validity, range accuracy, latency, and stationary jitter.")
    analyze.add_argument("--log", required=True, help="Run poses.jsonl.")
    analyze.add_argument("--stationary", action="store_true", help="Compute jitter; use only when both camera and tag were stationary.")
    analyze.add_argument("--distance", type=positive_float, help="Measured optical-center-to-tag-center distance in meters.")
    analyze.add_argument("--output", help="Save results to a new JSON file.")
    analyze.set_defaults(action=tracking.analyze)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("focus", "exposure"):
        value = getattr(args, name, None)
        if value is not None and not np.isfinite(value):
            parser.error(f"--{name} must be finite")
    try:
        args.action(args)
    except (ValueError, OSError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return 0
