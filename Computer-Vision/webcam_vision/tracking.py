"""Live/replayed video, overlays, raw recordings, and timestamped pose logs."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .calibration import load_calibration
from .common import camera_settings, check_size, open_camera, put_lines, require_gui, utc_now, write_json
from .pose import PosePolicy, TagTracker, rotation_distance


def annotate(frame, record, corners, ids, selected, tracker, fps, age_ms):
    image = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
    lines = [
        f"ID {tracker.target_id}: {record['status']} | {fps:.1f} updates/s",
        f"Processing {record['processing_ms']:.1f} ms | host age {age_ms:.1f} ms",
    ]
    if selected is not None:
        cv2.drawFrameAxes(image, tracker.estimator.matrix, tracker.estimator.distortion,
                         selected.rvec, selected.tvec, tracker.estimator.tag_size * 0.5, 2)
        position = record["camera_position_m"]
        orientation = record["camera_rpy_deg"]
        lines += [
            "Camera in TAG frame (m): " + " ".join(f"{axis}={value:+.3f}" for axis, value in zip("xyz", position)),
            f"Distance {record['distance_m']:.3f} m | RMS {record['reprojection_rms_px']:.2f} px",
            "Camera RPY (deg, ZYX): " + " ".join(f"{v:+.1f}" for v in orientation),
            "Tag axes: red X right / green Y up / blue Z out",
        ]
    else:
        lines.append("Camera position/orientation: UNAVAILABLE")
    lines.append("Q / ESC quit | poses are raw, not smoothed")
    put_lines(image, lines, (210, 255, 210) if record["valid"] else (100, 200, 255))
    return image


def read_timestamps(path):
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("frame_index") is None:
                continue
            if row["frame_index"] != len(result):
                raise ValueError("Timing log frame indexes must be consecutive from zero.")
            timestamp = float(row["timestamp_s"])
            if not np.isfinite(timestamp) or timestamp < 0 or (result and timestamp <= result[-1]):
                raise ValueError("Timing log timestamps must be finite, nonnegative, and strictly increasing.")
            result.append(timestamp)
    if not result:
        raise ValueError("Timing log contains no frames.")
    return result


def run(args):
    replay = args.command == "replay"
    calibration, matrix, distortion, size = load_calibration(args.calibration)
    if not args.headless:
        require_gui()
    tracker = TagTracker(matrix, distortion, args.tag_size, args.tag_id,
                         PosePolicy(max_reprojection_px=args.max_reprojection,
                                    ambiguity_gap_px=args.ambiguity_gap))
    timestamps = read_timestamps(args.timestamps) if replay and args.timestamps else None
    width, height = (args.width or size[0]), (args.height or size[1])
    if (width, height) != size:
        raise ValueError("Requested resolution differs from calibration. Recalibrate for the new mode.")
    source = str(Path(args.video).resolve()) if replay else args.camera
    if replay:
        if not Path(args.video).is_file():
            raise ValueError(f"Video does not exist: {args.video}")
        cap, warnings = cv2.VideoCapture(source), []
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open video: {args.video}")
    else:
        cap, warnings = open_camera(source, width, height, args.fps, args.focus, args.exposure)
    writer = None
    log = None
    output = None
    started = time.perf_counter()
    processed, valid_count, processing_total = 0, 0, 0.0
    end_reason = "completed"
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        if replay and timestamps is None and (not np.isfinite(source_fps) or source_fps <= 0):
            raise ValueError("Video has no usable frame rate; provide --timestamps from its recording.")
        output = Path(args.output or ("runs/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")))
        output.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema_version": 1, "created_utc": utc_now(), "opencv_version": cv2.__version__,
            "source": source, "mode": args.command, "calibration": calibration, **tracker.metadata(),
            "reported_camera_settings": camera_settings(cap), "warnings": warnings,
            "time_basis": "recording_log" if timestamps else "frame_index_over_fps" if replay else "host_receive_monotonic",
            "timestamp_note": "Host timestamps are read-completion times, not sensor exposure times. Processing excludes capture buffering.",
            "frames": {"tag": "x decoded right, y up, z out of printed face", "camera": "x right, y down, z forward"},
            "transform_note": "T_a_from_b maps coordinates from frame b into frame a; translations in meters.",
            "raw_video": "camera.avi" if args.record else None,
            "recording_note": "AVI uses nominal constant FPS. Replay with --timestamps poses.jsonl to preserve observation intervals.",
        }
        write_json(output / "session.json", metadata)
        log = (output / "poses.jsonl").open("x", encoding="utf-8")
        for warning in warnings:
            print(f"Warning: {warning}")
        print(f"Logging to {output}. " + ("Headless mode; Ctrl+C to stop." if args.headless else "Q / ESC to stop."))
        first_receive = None
        previous_tick = None
        update_rate = 0.0
        while True:
            ok, frame = cap.read()
            received = time.perf_counter()
            received_utc = utc_now()
            if not ok:
                if replay:
                    # Backends generally report both EOF and decode failures as read=False.
                    expected_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if processed == 0 or (expected_count > 0 and processed < expected_count):
                        raise ValueError("Video ended before its reported frame count (decode error or truncation).")
                    if timestamps is not None and processed != len(timestamps):
                        raise ValueError("Video frame count does not match the supplied timing log.")
                    end_reason = "end_of_video"
                    break
                raise ValueError("Camera stopped delivering frames; pose is unavailable.")
            check_size(frame, size)
            if first_receive is None:
                first_receive = received
            if replay:
                if timestamps is not None:
                    if processed >= len(timestamps):
                        raise ValueError("Video has more frames than the supplied timing log.")
                    timestamp = timestamps[processed]
                else:
                    timestamp = processed / source_fps
                if not args.headless:
                    first_timestamp = timestamps[0] if timestamps else 0.0
                    delay = timestamp - first_timestamp - (time.perf_counter() - first_receive)
                    # Bounded waits keep replay responsive even across a long recording gap.
                    while delay > 0:
                        if cv2.waitKey(max(1, min(50, int(delay * 1000)))) & 0xFF in (ord("q"), 27):
                            end_reason = "user_quit"
                            return output
                        delay = timestamp - first_timestamp - (time.perf_counter() - first_receive)
            else:
                timestamp = received - first_receive
            tick = time.perf_counter()
            if previous_tick is not None and tick > previous_tick:
                instantaneous = 1 / (tick - previous_tick)
                update_rate = instantaneous if update_rate == 0 else 0.9 * update_rate + 0.1 * instantaneous
            previous_tick = tick
            record, corners, ids, selected = tracker.process(frame, timestamp)
            elapsed_ms = (time.perf_counter() - tick) * 1000
            record.update(schema_version=1, frame_index=processed, received_utc=received_utc,
                          processing_ms=elapsed_ms, update_rate_hz=update_rate)
            if args.record:
                if writer is None:
                    nominal_fps = source_fps if np.isfinite(source_fps) and source_fps > 0 else args.fps
                    writer = cv2.VideoWriter(str(output / "camera.avi"), cv2.VideoWriter_fourcc(*"MJPG"), nominal_fps, size)
                    if not writer.isOpened():
                        raise ValueError("Cannot initialize MJPG video recording. Retry without --record.")
                writer.write(frame)  # Raw, without overlays, so detection can be replayed.
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            processed += 1
            valid_count += int(record["valid"])
            processing_total += elapsed_ms
            if not args.headless:
                display = annotate(frame, record, corners, ids, selected, tracker, update_rate,
                                   (time.perf_counter() - received) * 1000)
                cv2.imshow("AprilTag camera pose", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    end_reason = "user_quit"
                    break
            if args.max_frames and processed >= args.max_frames:
                end_reason = "max_frames"
                break
    except (ValueError, OSError, cv2.error) as error:
        end_reason = str(error)
        if log is not None:
            log.write(json.dumps({"schema_version": 1, "frame_index": None, "valid": False,
                                  "status": "source_error", "message": str(error), "received_utc": utc_now(),
                                  "T_tag_from_camera": None, "camera_position_m": None, "distance_m": None}) + "\n")
        raise
    except KeyboardInterrupt:
        end_reason = "interrupted"
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if log is not None:
            log.close()
        if not args.headless:
            cv2.destroyAllWindows()
        if output is not None and log is not None:
            duration = time.perf_counter() - started
            summary = {"frames": processed, "valid_frames": valid_count, "elapsed_s": duration,
                       "loop_throughput_hz": processed / max(duration, 1e-9),
                       "mean_processing_ms": processing_total / processed if processed else None,
                       "end_reason": end_reason}
            write_json(output / "summary.json", summary)
            print(json.dumps(summary, indent=2))
    return output


def analyze(args):
    with Path(args.log).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    frames = [row for row in rows if row.get("frame_index") is not None]
    valid = [row for row in frames if row["valid"]]
    status_counts = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    result = {"frames": len(frames), "valid_frames": len(valid), "status_counts": status_counts}
    if frames:
        processing = [row["processing_ms"] for row in frames]
        duration = frames[-1]["timestamp_s"] - frames[0]["timestamp_s"]
        result.update(mean_processing_ms=float(np.mean(processing)), p95_processing_ms=float(np.percentile(processing, 95)),
                      observation_rate_hz=(len(frames) - 1) / duration if duration > 0 else None,
                      valid_fraction=len(valid) / len(frames))
    if valid:
        distances = np.array([row["distance_m"] for row in valid])
        result["median_distance_m"] = float(np.median(distances))
        result["median_reprojection_rms_px"] = float(np.median([row["reprojection_rms_px"] for row in valid]))
        if args.distance is not None:
            errors = np.abs(distances - args.distance) / args.distance * 100
            result.update(reference_distance_m=args.distance, median_absolute_range_error_percent=float(np.median(errors)),
                          p95_absolute_range_error_percent=float(np.percentile(errors, 95)))
        if args.stationary:
            positions = np.array([row["camera_position_m"] for row in valid])
            rotations = [np.asarray(row["T_tag_from_camera"])[:3, :3] for row in valid]
            deviations = np.degrees([rotation_distance(rotations[0], r) for r in rotations])
            # Only compare consecutive valid frames, never bridge a tracking loss.
            steps = [np.degrees(rotation_distance(rotations[i - 1], rotations[i]))
                     for i in range(1, len(valid)) if valid[i]["frame_index"] == valid[i - 1]["frame_index"] + 1]
            result.update(position_std_xyz_m=np.std(positions, axis=0).tolist(),
                          orientation_p95_deviation_from_first_deg=float(np.percentile(deviations, 95)),
                          max_consecutive_orientation_step_deg=float(max(steps)) if steps else None)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.output:
        write_json(args.output, result)
    return result
