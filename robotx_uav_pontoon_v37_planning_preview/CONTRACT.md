# RobotX UAV Pontoon -> Planning contract (preview v0.1)

## ROS2 topics

- Primary: `/robotx/perception/uav/pontoon/observation`
  `robotx_uav_pontoon_interfaces/msg/PontoonObservation` (RELIABLE, depth 10).
- Human-readable heartbeat: `/robotx/perception/uav/pontoon/status`
  `std_msgs/msg/String` (JSON). **Never** drive control from this debug string.

## Coordinate conventions

- `header.stamp`: **original RGB image acquisition stamp** (not publish time).
- `header.frame_id`: actual camera optical frame (+X right, +Y down, +Z forward).
- `*_uv_px`: image pixels, (0,0) upper-left, x=u right, y=v down, **original input image dimensions**.
- `*_world_m`: world frame in `world_frame_id`, default `map_ned` (+X north,+Y east,+Z down), meters.
- `bbox_xywh_px`: top-left x/y and width/height in source-image pixels.
- `geometry_score`: shape fitting score, **NOT calibrated confidence/probability**.
- `branch.index` 1..4 and label B1..B4 are **image-order temporary labels**; no physical identity is guaranteed under 180-degree viewpoint ambiguity.

## Validity and safety contract

- `geometry_ready`: FULL_PLATFORM, detected, layout verified, and all 4 sets of tip/root/center and trunk points in bounds.
- `xyz_valid`: all 15 points (3 trunk + 4*3 branches) have metric coordinates in world frame. If false, do not use world points (they are zero placeholders).
- `branch_ids_verified`: external physical identity check explicitly approved by operator; false by default.
- `uv_coordinates_verified`: operator confirmed detector UV coordinates refer to the ORIGINAL RGB image resolution. False by default.
- `metric_calibration_verified`: operator measured real XYZ error with surveyed targets and accepted it. False by default.
- `planning_ready`: `geometry_ready && xyz_valid && branch_ids_verified && uv_coordinates_verified && metric_calibration_verified`.
- `invalid_reason` explains a failure. Planner must additionally check **freshness** using current ROS time minus header.stamp, recommended <=0.5 s, and must not assume an observation is a safe flight/landing waypoint.
- Never fly or dispense solely on a single perception frame; planner owns trajectory safety, geofencing, collision checks, and multi-frame confirmation.

## Metric projection methods

1. `aligned_depth`: pixel-aligned 16UC1 (millimeters) or 32FC1 (meters) depth. With `input_rectified=true` (default), rays use CameraInfo.P. With `input_rectified=false`, raw RGB rays use CameraInfo.K/D with OpenCV's pinhole `undistortPoints` model. Depth must match the actual input RGB pixels, image size, optical frame, and timestamp within 80 ms. A 5x5 median patch is used; depth edges with >0.30 m 10th-to-90th percentile spread are rejected.
2. `known_plane`: optional explicit measured top-plane height `plane_z_ned_m`, verified and enabled. Pixel rays intersect that plane using timestamped TF. This is **not** generic monocular depth and is invalid when pontoon top is not horizontal/at the configured height.
3. `mixed`: different points obtained by different methods; requires all points valid to set `xyz_valid=true`.
4. `none`: no valid metric projection.

Do not enable known-plane projection using a guessed height. Never publish a fabricated static `map_ned -> camera` TF just to make `xyz_valid` true.

## Dependencies to verify before real planning

- Detector's exact V3.7 result schema (run `scripts/probe_v37.py`); the adapter fails closed if not mapped.
- Input uses the configured raw/rectified camera model, and detector UV points correspond to source image pixel coordinates. Raw depth must be registered to raw RGB pixels; matching topic names alone do not establish alignment.
- Real TF chain at image timestamp: `map_ned -> uav_base_link -> uav_camera_mount -> uav_camera_optical_frame`.
- Actual UAV source topics and depth alignment (if any).
- Measured plane height or calibrated depth, plus ground truth comparison with tape measure / surveyed targets.
- Physical branch B1..B4 numbering, including opposite UAV heading and 180-degree rotation.

## Relationship to existing RobotX resources contract

This package publishes its own typed `pontoon/observation` topic and does not modify `/robotx/perception/uav/resources` or `/robotx/perception/uav/resources_world`. The planning team can build a bridge to those existing contracts after agreeing on what a pontoon branch represents. Do not silently repurpose the existing `TargetObservation` fields.

## Mission control and installed detector

`source_dir=''` loads the V3.7 detector and its dependencies installed under this package's `share/robotx_uav_pontoon_bridge/detector`; an explicit directory remains supported. No developer-specific path is required.

Standalone behavior remains enabled by default. With `mission_control.enabled=true`, the node starts OFF and subscribes to `mission_control.topic` (default `/tongji_mrl/tsky/downward_vision_control`) using reliable, transient-local QoS, depth 1. JSON commands have `mode` and an `object_types` list. Detection runs only for `{"mode":"MAP","object_types":["dock"]}` (other selected object types may coexist). Mode/selection changes preserve buffered depth and keep the frame-local detector loaded. Depth reuse follows the existing image-size, optical-frame and timestamp checks. This control does not change any metric or planning validity checks.
