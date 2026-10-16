# Webcam calibration and AprilTag pose tracking

Estimate an **unmounted webcam's position and orientation relative to a stationary AprilTag**. No robot connection or machine learning is needed. The first demo is moving the webcam around a fixed tag and watching its pose update.

## Setup

Use Python 3.11. From this directory, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe -m webcam_vision --help
```

The local `.venv` may already be prepared. Activation is optional; the commands below explicitly use its interpreter. On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

Install only one OpenCV distribution in the environment: this project uses **opencv-contrib-python**, which includes `aruco` and desktop windows. Avoid mixing it with `opencv-python` or either headless package. `pip install -e .` is also supported if exact tested versions are not needed.

## 1. Print and measure targets

```powershell
.\.venv\Scripts\python.exe -m webcam_vision targets --output targets --tag-id 0
```

This generates `tag36h11_0.png`, `checkerboard.png`, and target metadata. The default checkerboard has **9 columns × 6 rows of internal corners**, meaning **10 × 7 squares**. Print without stretching and mount both patterns flat on rigid backing. Keep the tag's white margin visible. Avoid glossy surfaces and screen displays for final calibration.

- Aim for a tag about **150 mm** wide. Measure the printed **outer black marker width**, excluding its white margin. Convert millimeters to meters: 150 mm = `0.150` m.
- Measure one checkerboard square, ideally averaging several adjacent squares. Example: 25 mm = `0.025` m.
- Image pixel dimensions do not specify print dimensions. Printer scaling can change the actual measurements. A 25 mm-square board may need a larger sheet; a smaller board is fine when measured correctly.
- The generated image's top defines the tag's decoded top. Rotating it physically rotates the reference axes with it.

Use your measured values in subsequent commands; **the example dimensions below are not assumed measurements**.

## 2. Capture calibration images

```powershell
.\.venv\Scripts\python.exe -m webcam_vision capture --camera 0 --width 1280 --height 720 --columns 9 --rows 6 --square-size 0.025 --output data/calibration-session
```

| Key | Action |
|---|---|
| Space | Save a training image, only when all board corners are detected |
| V | Save an independent held-out validation image |
| Q / Escape | Finish and save capture metadata |

Collect **30–50 training images and 5–10 held-out images**. Move and tilt the board between captures; include close/far views and all image regions. Keep corners sharp, well-lit, and fully visible. Do not fill the dataset with nearly identical frames.

Keep the capture resolution, focus, and zoom fixed during calibration and tracking. Disable autofocus in the camera's controls if possible. `--focus VALUE` requests manual focus using device-specific units, but driver acknowledgments/readbacks do not guarantee it worked. `--exposure VALUE` requests exposure; select manual exposure mode externally because backend mode values vary. Verify settings yourself, especially after reconnecting.

The tool verifies actual frame dimensions; it fails rather than silently resizing if the camera refuses the requested mode. Choose another supported resolution and use it consistently. Recalibrate when focus, zoom, cropping, or resolution changes. Camera motion alone does not require new intrinsics.

## 3. Fit intrinsics and validate

```powershell
.\.venv\Scripts\python.exe -m webcam_vision calibrate --images data/calibration-session/train --validation data/calibration-session/validation --columns 9 --rows 6 --square-size 0.025 --output calibration.json --previews data/calibration-previews
```

At least 10 detectable training views and 3 held-out views are required. The tool rejects mixed resolutions, reuse of the same training/validation folder, and byte-identical images shared between the sets. It reports unreadable images or missing boards in `skipped_images` rather than silently hiding them.

`calibration.json` contains:

- Camera matrix `K` (`fx`, `fy`, `cx`, `cy`) and radial/tangential distortion coefficients.
- Image size `[width, height]`, measured board geometry, timestamp, and OpenCV version.
- Training and held-out reprojection RMS, per-view errors, and capture settings when available.

The held-out test keeps intrinsics fixed and fits each validation board's pose. RMS is `sqrt(mean(dx² + dy²))` over corners, in pixels. It tests projection consistency, **not absolute distance accuracy**. Inspect the original/undistorted previews for curved straight lines and edge errors. A high-error calibration is saved with a warning for inspection; it is not automatically declared successful. Physical validation is still required.

This uses a pinhole model with standard lens distortion. A strongly fisheye lens may need a different calibration model, which is outside V1.

## 4. Track camera pose and extrinsics

Fix the tag to a stationary surface with clear lighting. Start slightly off-axis to help resolve planar ambiguity:

```powershell
.\.venv\Scripts\python.exe -m webcam_vision track --camera 0 --calibration calibration.json --tag-id 0 --tag-size 0.150 --output runs/first-demo --record
```

The tracker requests the calibrated resolution by default. It overlays camera position, distance, orientation, tag axes, status, processing time, and update rate. Press Q/Escape to exit. `--headless` logs without a display; `--max-frames 300` limits a run. New output paths are required; existing data is never overwritten.

### What the extrinsics mean

**Intrinsics** describe the lens and image geometry. **Extrinsics** describe the pose relative to a reference frame. Here that reference is the tag, so moving the camera changes its extrinsics every frame. No camera-to-robot calibration is needed until the camera is mounted on a robot.

| Frame | Origin | X | Y | Z |
|---|---|---|---|---|
| Tag | Tag center | Decoded right | Decoded up | Out of the printed face |
| Camera | Optical center | Image right | Image down | Forward along viewing direction |

`T_a_from_b` maps a point in frame `b` into frame `a`. OpenCV estimates `T_camera_from_tag`; we invert it to obtain the camera pose `T_tag_from_camera`:

```text
R_tag_from_camera = R_camera_from_tag.T
t_tag_from_camera = -R_camera_from_tag.T @ t_camera_from_tag
```

If the camera is directly in front of an upright tag, its tag-frame Z is positive. Its rotation is not identity: camera Y points down and its viewing Z points toward the tag. Position is in meters. Distance is the Euclidean camera-optical-center-to-tag-center distance, not simply the camera's Z coordinate.

The on-image axes belong to the **tag**: red X, green Y, blue Z. Camera orientation is displayed as roll/pitch/yaw in degrees using `R = Rz(yaw) Ry(pitch) Rx(roll)`. Euler angles can wrap or become singular; use the logged rotation matrix for computation. Neither coordinate frame is automatically gravity-aligned or room-aligned. A moving tag remains a valid relative reference but cannot measure camera motion relative to a fixed room.

### Pose quality and ambiguity

The detector uses `36h11` and decoded TL/TR/BR/BL corners, matching `SOLVEPNP_IPPE_SQUARE` object-point order. It estimates poses from **original distorted pixels plus K and distortion**, avoiding double correction.

Raw candidates are rejected if non-finite, any corner is behind the camera, or the camera is behind the printed front face. Minimum image side length is 12 pixels; maximum accepted reprojection RMS is 2.5 pixels (`--max-reprojection`).

If candidate errors differ by at most 0.25 pixels (`--ambiguity-gap`), materially different poses require distinguishing recent history. Near-identical solutions within 2 degrees and 2 mm are treated as equivalent. History expires after 0.25 seconds. For an ambiguous pair, candidates must be within 0.5 m and 45 degrees of the previous pose, and continuity costs must differ by at least 0.25 (`translation/tag_size + angle/30deg`). Otherwise the frame is invalid. History resets after any invalid frame or target loss. This is a heuristic, not a calibrated confidence probability.

Far-away or nearly frontal tags can be ambiguous; move closer, tilt the view slightly, or use a larger tag. Reprojection error alone is not proof of correct pose. No smoothing is applied, so raw jitter remains visible for validation. Duplicate copies of the selected ID are invalid rather than arbitrarily switching references.

## 5. Record, replay, and inspect results

Each run writes:

| File | Contents |
|---|---|
| `session.json` | Full calibration snapshot, source, tag settings, coordinate conventions, thresholds, camera readbacks, and timestamp semantics |
| `poses.jsonl` | One record per processed frame: timestamp, received UTC, frame index, detected IDs, validity/reason, corners, both transforms, camera position/RPY, distance, reprojection RMS, raw candidates, and processing time |
| `summary.json` | Counts, loop throughput, mean processing time, and termination reason |
| `camera.avi` | Optional raw MJPG frames without overlays, suitable for replay |

Invalid records have **null accepted poses and distance**; raw hypotheses can remain for diagnosis. A camera read failure adds an invalid terminal `source_error` event and exits with a nonzero status. Consumers must also check freshness and process health; a last valid record must not be treated as live indefinitely.

Live timestamps are monotonic seconds since the first frame was received; UTC is also recorded. They are **host read-completion timestamps**, not sensor exposure timestamps. Processing latency excludes driver buffering. V1 cannot measure full exposure-to-display latency, and a blocking camera driver may delay reporting a disconnect.

AVI stores nominal constant FPS, whereas webcam delivery may vary. Use the original log when replaying recordings to restore actual observation intervals:

```powershell
.\.venv\Scripts\python.exe -m webcam_vision replay --video runs/first-demo/camera.avi --timestamps runs/first-demo/poses.jsonl --calibration calibration.json --tag-size 0.150 --output runs/replay
```

Without `--timestamps`, replay uses frame index divided by the video's FPS. Displayed replay follows source timing; `--headless` processes as fast as possible. Calibration resolution must also match recorded video. MJPG is lossy, so detections may differ slightly from the original stream.

For a run recorded with **both camera and tag stationary**, at a measured distance of 1 m:

```powershell
.\.venv\Scripts\python.exe -m webcam_vision analyze --log runs/first-demo/poses.jsonl --stationary --distance 1.0 --output runs/first-demo/analysis.json
```

This reports validity fraction, status counts, median/p95 absolute range errors, position standard deviation, orientation deviations, and processing latency. `--stationary` is an assertion about your experiment; omit it for moving-camera runs. `--distance` also assumes the reference distance stayed constant. Report valid fraction alongside accuracy; invalid frames are excluded from pose statistics.

## Validation checklist

These are **initial project targets**, not promised hardware performance. Synthetic tests do not certify a physical webcam.

| Area | Procedure | Target / recorded result |
|---|---|---|
| Intrinsics | Evaluate independent checkerboard images; inspect corrected edges | Held-out reprojection RMS <1 pixel; no obvious residual curvature |
| Distance | Separate stationary recordings at 0.5, 1, and 2 m; measure optical-center-to-tag-center distance | Aim for <=5% range error; report median, p95, valid fraction, and ruler uncertainty |
| Coordinates | Move camera in known directions while tag stays fixed | Position signs match the defined tag axes |
| Stationary stability | Record at least 30 seconds at each distance | Report XYZ standard deviations, rotation deviations, and largest consecutive rotation step; investigate flips |
| Viewing envelope | Increase tilt and test positions near image edges | Record usable range/angles and validity; reject unreliable poses |
| Loss/identity | Cover target, show wrong ID, or show two copies of target ID | Pose unavailable, no stale accepted position |
| Device/configuration | Disconnect webcam; request mismatched resolution | Clear error and invalid/terminated tracking |
| Runtime | Run at calibrated resolution | Aim for >=15 updates/s; report actual throughput and processing latency |
| Mathematics | Synthetic projected corners, transform inversion, and rotated printed patterns | Recover known transforms within numerical/rasterization tolerance |

Run the automated suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover distorted synthetic geometry, decoded tag orientation, ambiguity/history, duplicates, disappearance, calibration fitting on independent synthetic views, target generation, video recording/replay, analysis, and a simulated camera disconnect.

## Future vision and robot learning work

| Project | What it enables | Additional requirements | Difficulty |
|---|---|---|---|
| Onboard AprilTag following | Turn toward a moving target and maintain distance | Rigid mount, camera-to-body extrinsics, walking-command adapter, stale-observation stop | Low–medium |
| Tag docking and waypoint routes | Approach a station or follow marked destinations | Relative-position controller; surveyed locations for mapped routes | Medium |
| Fixed-tag room localization | Estimate robot position/orientation in a room | Known tag poses and camera-to-body extrinsics; multiple tags for coverage | Medium |
| Learning evaluation with tags | Measure trajectories, tracking error, and success | Fixed tags, synchronized logs, visibility-gap handling | Low–medium |
| Imitation learning for steering | Learn high-level navigation from demonstrations | Images paired with operator commands/state; held-out recording sessions | Medium |
| Person or object following | Follow without printed markers | Detector, tracker, and reliable distance estimation | Medium |
| Visual-inertial localization | Track motion between tag sightings | Synchronized camera/IMU, sensor calibration, and localization software | High |
| Terrain recognition and gait selection | Choose speeds or existing gaits for surfaces | Labeled terrain images and gait choices; appearance alone does not establish traversability | Medium–high |
| Perception-guided footholds/locomotion | Anticipate steps and uneven ground | Prefer depth/stereo, proprioception, simulation, and an established walking controller | High |

Recommended progression: **standalone tracking → onboard following and synchronized logging → fixed-tag localization and imitation learning for steering → terrain-aware locomotion**. Preserve calibration, settings, timestamps, and images now; add synchronized commands, IMU, and joint states when integrating the robot. Monocular vision without known size or another metric reference does not provide absolute scale.

## References

- [OpenCV camera calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [OpenCV marker detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [OpenCV PnP and IPPE corner order](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
- [AprilTag project and tag-size convention](https://github.com/AprilRobotics/apriltag)
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [Perceptive quadruped locomotion](https://leggedrobotics.github.io/rl-perceptiveloco/)
