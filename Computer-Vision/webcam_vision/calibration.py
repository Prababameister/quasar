"""Checkerboard collection, camera calibration, and held-out evaluation."""

import json
from pathlib import Path

import cv2
import numpy as np

from .common import camera_settings, check_size, open_camera, put_lines, require_gui, utc_now, write_json


def board_points(columns, rows, square_size):
    if columns < 3 or rows < 3 or not np.isfinite(square_size) or square_size <= 0:
        raise ValueError("Board needs at least 3x3 internal corners and a positive square size in meters.")
    points = np.zeros((columns * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size
    return points


def find_board(gray, columns, rows):
    found, corners = cv2.findChessboardCornersSB(
        gray, (columns, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    return corners.astype(np.float32) if found else None


def reprojection_rms(points, corners, rvec, tvec, matrix, distortion):
    projected, _ = cv2.projectPoints(points, rvec, tvec, matrix, distortion)
    residuals = projected.reshape(-1, 2) - np.asarray(corners).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))


def load_calibration(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported calibration schema; generate a calibration with this application.")
    try:
        matrix = np.asarray(value["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(value["distortion_coefficients"], dtype=np.float64).reshape(-1)
        size = value["image_size"]
        valid = (
            matrix.shape == (3, 3) and np.all(np.isfinite(matrix))
            and matrix[0, 0] > 0 and matrix[1, 1] > 0
            and np.allclose(matrix[2], [0, 0, 1])
            and distortion.size in (4, 5, 8, 12, 14) and np.all(np.isfinite(distortion))
            and len(size) == 2 and all(isinstance(x, int) and x > 0 for x in size)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Malformed calibration file.") from error
    if not valid:
        raise ValueError("Calibration contains invalid camera parameters or image dimensions.")
    return value, matrix, distortion, tuple(size)


def capture(args):
    require_gui()
    board_points(args.columns, args.rows, args.square_size)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "train").mkdir()
    (folder / "validation").mkdir()
    cap, warnings = open_camera(args.camera, args.width, args.height, args.fps, args.focus, args.exposure)
    for warning in warnings:
        print(f"Warning: {warning}")
    counts = {"train": 0, "validation": 0}
    metadata = {
        "schema_version": 1, "created_utc": utc_now(),
        "board": {"columns": args.columns, "rows": args.rows, "square_size_m": args.square_size},
        "requested_image_size": [args.width, args.height], "camera_index": args.camera,
        "reported_camera_settings": camera_settings(cap), "warnings": warnings,
        "note": "Reported camera controls may be unsupported; verify focus/zoom manually.",
        "frames": [],
    }
    print("SPACE: save training view | V: save held-out view | Q: quit. Move/tilt board between saves.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Camera stopped delivering frames during calibration capture.")
            check_size(frame, (args.width, args.height))
            corners = find_board(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), args.columns, args.rows)
            display = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(display, (args.columns, args.rows), corners, True)
            put_lines(display, [
                f"Board: {'FOUND' if corners is not None else 'not found'}",
                f"Train: {counts['train']}   Held-out: {counts['validation']}",
                "SPACE train | V held-out | Q quit",
            ])
            cv2.imshow("Calibration capture", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            subset = "train" if key == 32 else "validation" if key == ord("v") else None
            if subset and corners is not None:
                name = f"{subset}/{counts[subset]:04d}.png"
                if not cv2.imwrite(str(folder / name), frame):
                    raise ValueError(f"Could not save {name}.")
                counts[subset] += 1
                metadata["frames"].append({"file": name, "received_utc": utc_now()})
                print(f"Saved {name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        metadata["counts"] = counts
        write_json(folder / "capture.json", metadata)


def collect_views(folder, columns, rows, expected_size=None):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Image directory does not exist: {folder}")
    views, skipped = [], []
    size = expected_size
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            skipped.append({"file": str(path), "reason": "unreadable"})
            continue
        current_size = (gray.shape[1], gray.shape[0])
        if size is None:
            size = current_size
        if current_size != size:
            raise ValueError(f"Mixed image resolutions: {path} is {current_size}, expected {size}.")
        corners = find_board(gray, columns, rows)
        if corners is None:
            skipped.append({"file": str(path), "reason": "board_not_found"})
        else:
            views.append((str(path.resolve()), corners))
    return views, skipped, size


def calibrate(args):
    if Path(args.output).exists():
        raise ValueError(f"Output already exists: {args.output}. Choose another name to preserve it.")
    if Path(args.images).resolve() == Path(args.validation).resolve():
        raise ValueError("Training and held-out directories must be different.")
    points = board_points(args.columns, args.rows, args.square_size)
    training, skipped_train, size = collect_views(args.images, args.columns, args.rows)
    validation, skipped_validation, _ = collect_views(args.validation, args.columns, args.rows, size)
    if len(training) < 10:
        raise ValueError(f"Only {len(training)} usable training views; need at least 10, recommend 30-50.")
    if len(validation) < 3:
        raise ValueError(f"Only {len(validation)} usable held-out views; need at least 3, recommend 5-10.")
    # Catch identical copied images as well as overlapping paths, to keep the holdout meaningful.
    import hashlib
    training_hashes = {hashlib.sha256(Path(path).read_bytes()).digest() for path, _ in training}
    if any(hashlib.sha256(Path(path).read_bytes()).digest() in training_hashes for path, _ in validation):
        raise ValueError("A held-out image is identical to a training image. Capture independent views.")
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        [points] * len(training), [corners for _, corners in training], size, None, None
    )
    if not np.isfinite(rms) or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(distortion)):
        raise ValueError("Calibration produced non-finite parameters; check board dimensions and views.")
    train_errors = [
        {"file": path, "rms_px": reprojection_rms(points, corners, r, t, matrix, distortion)}
        for (path, corners), r, t in zip(training, rotations, translations)
    ]
    validation_errors = []
    for path, corners in validation:
        ok, rvec, tvec = cv2.solvePnP(points, corners, matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError(f"Could not estimate board pose for held-out view {path}.")
        validation_errors.append({
            "file": path, "rms_px": reprojection_rms(points, corners, rvec, tvec, matrix, distortion)
        })
    validation_rms = float(np.sqrt(np.mean([row["rms_px"] ** 2 for row in validation_errors])))
    capture_metadata = None
    capture_path = Path(args.images).parent / "capture.json"
    if capture_path.exists():
        capture_metadata = json.loads(capture_path.read_text(encoding="utf-8"))
        if capture_metadata.get("board") != {"columns": args.columns, "rows": args.rows, "square_size_m": args.square_size}:
            raise ValueError("Board arguments differ from capture.json. Use the recorded board measurements.")
    result = {
        "schema_version": 1, "created_utc": utc_now(), "opencv_version": cv2.__version__,
        "image_size": list(size), "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(), "model": "pinhole_radial_tangential",
        "board": {"columns": args.columns, "rows": args.rows, "square_size_m": args.square_size},
        "training_rms_px": float(rms), "held_out_rms_px": validation_rms,
        "training_views": train_errors, "held_out_views": validation_errors,
        "skipped_images": skipped_train + skipped_validation, "capture_metadata": capture_metadata,
        "held_out_under_1px": validation_rms < 1.0,
    }
    write_json(args.output, result)
    print(f"Saved {args.output}: training RMS {rms:.3f}px; held-out RMS {validation_rms:.3f}px.")
    print(f"Used {len(training)} training / {len(validation)} held-out; skipped {len(result['skipped_images'])}.")
    if validation_rms >= 1:
        print("Warning: held-out RMS misses the 1px target. Inspect per-view errors and capture better views.")
    if args.previews:
        previews = Path(args.previews)
        previews.mkdir(parents=True, exist_ok=False)
        for index, (path, _) in enumerate(validation):
            original = cv2.imread(path)
            corrected = cv2.undistort(original, matrix, distortion)
            compare = np.hstack((original, corrected))
            put_lines(compare, ["Original (left) | Undistorted (right)"])
            if not cv2.imwrite(str(previews / f"{index:04d}.png"), compare):
                raise ValueError("Could not save validation preview.")
