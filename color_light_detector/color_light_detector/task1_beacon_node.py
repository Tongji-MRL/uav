#!/usr/bin/env python3
"""RobotX Task 1 production multi-beacon perception node."""

import json
import math
import time
from uuid import uuid4
from pathlib import Path
from typing import Dict, List

import cv2
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics.cfg import DEFAULT_CFG_DICT

from .aperture_photometry import DEFAULTS as APERTURE_DEFAULTS
from .emitter_fusion import DEFAULTS as EMITTER_FUSION_DEFAULTS
from .roi_motion import RoiMotion
from .risk_scheduler import RiskScheduler, DEFAULTS as RISK_DEFAULTS
from .roi_continuity import RoiContinuity, DEFAULTS as ROI_CONTINUITY_DEFAULTS
from .scheduled_detector import ScheduledDetector, CONTROL_DEFAULTS, validate_control
from .online_light import OnlineLightReference, DEFAULTS as ONLINE_LIGHT_DEFAULTS
from .light_reference_bank import LightReferenceBank
from .lamp_region import LampRegionLocator, DEFAULTS as LAMP_REGION_DEFAULTS
from .lamp_photometry import LampPhotometer, DEFAULTS as LAMP_PHOTOMETRY_DEFAULTS
from .plane_localization_ros import PlaneLocalizationROS
from .engine_geometry import engine_contract
from .model_output import load_model, detections_from_prediction
from .mission_control import MissionControl
from .native_stream_status import NativeStreamStatus, status_text
from .local_zoom import (InferenceRegionPlanner, DEFAULTS as LOCAL_ZOOM_DEFAULTS,
                         validate as validate_zoom, fit_canvas, add_roi_panel)

from .task1_beacon_core import (
    TemporalResult,
    Detection,
    HsvColorClassifier,
    MultiBeaconTracker,
    Track,
    analysis_bbox,
    update_missing_track,
    update_track_observation,
)


class Task1BeaconNode(Node):
    """Detect, track and classify every visible Task 1 light beacon."""

    def __init__(self, enable_subscription: bool = True, *, parameter_overrides=None, context=None) -> None:
        super().__init__("task1_beacon", parameter_overrides=parameter_overrides, context=context)
        self.last_payload = None
        self.detector_worker = None
        self._declare_parameters()
        self.mission_control = MissionControl(
            self._value("mission_control.enabled"), "buoy", ("MAP", "TASK1"))

        model_name = str(self._value("model_path"))
        if not model_name:
            from ament_index_python.packages import get_package_share_directory
            model_name = str(Path(get_package_share_directory("color_light_detector")) /
                             "weights/best.pt")
        model_path = Path(model_name).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self.image_topic = str(self._value("image_topic"))
        self.algorithm_version = str(self._value("algorithm_version"))
        self.confidence = float(self._value("confidence"))
        self.image_size = int(self._value("image_size"))
        self.zoom_parameters = validate_zoom(self._parameters_with_prefix("local_zoom."))
        self.region_planner = InferenceRegionPlanner()
        self.last_inference_region = {"kind": "WAITING"}
        cpu_threads = int(self._value("processing.cpu_threads"))
        if not 1 <= cpu_threads <= 8:
            raise ValueError("processing.cpu_threads must be 1..8")
        cv2.setNumThreads(cpu_threads)
        torch.set_num_threads(cpu_threads)
        self.inference_precision = int(self._value("inference_precision"))
        if self.inference_precision not in (16, 32):
            raise ValueError("inference_precision must be 16 or 32")
        self.max_detections = int(self._value("max_detections"))
        self.max_fps = float(self._value("max_fps"))
        self.roi_padding_ratio = float(self._value("roi_padding_ratio"))
        self.minimum_confirmed_hits = int(
            self._value("minimum_confirmed_hits")
        )
        self.lit_confirm_frames = max(
            1, int(self._value("color.lit_confirm_frames"))
        )
        self.dark_confirm_frames = max(
            1, int(self._value("color.dark_confirm_frames"))
        )
        self.color_switch_frames = max(
            1, int(self._value("color.color_switch_frames"))
        )
        self.off_without_lit_min_hits = max(
            1, int(self._value("color.off_without_lit_min_hits"))
        )
        self.off_without_lit_min_detector_confidence = float(
            self._value("color.off_without_lit_min_detector_confidence")
        )
        self.emitter_hint_update_confidence_min = float(
            self._value(
                "color.emitter_hint_update_confidence_min"
            )
        )
        self.emitter_hint_update_alpha = float(
            self._value("color.emitter_hint_update_alpha")
        )
        self.debug_topic = str(self._value("debug_topic"))
        self.debug_max_fps = float(self._value("debug_max_fps"))
        self.debug_always_publish = bool(
            self._value("debug_always_publish")
        )
        self.debug_show_missing_tracks = bool(
            self._value("debug_show_missing_tracks")
        )
        debug_size = list(self._value("debug_image_size"))
        if len(debug_size) != 2:
            raise ValueError("debug_image_size must be [width, height]")
        if debug_size == [0, 0]:
            self.debug_image_size = None
        elif min(debug_size) <= 0:
            raise ValueError(
                "debug_image_size must be positive or [0, 0]"
            )
        else:
            self.debug_image_size = (
                int(debug_size[0]),
                int(debug_size[1]),
            )
        self.result_topic = str(self._value("result_topic"))
        self.association_mode = str(
            self._value("tracking.association_mode")
        )
        self.color_roi_bbox_source = str(
            self._value("tracking.color_roi_bbox_source")
        )
        self.temporal_color_evidence_mode = str(
            self._value("temporal_color_evidence_mode")
        )
        self.state_change_enabled = bool(
            self._value("state_change.enabled")
        )
        self.high_confidence_threshold = float(
            self._value("tracking.high_confidence_threshold")
        )
        self.low_confidence_threshold = float(
            self._value("tracking.low_confidence_threshold")
        )
        self.inference_confidence = (
            self.low_confidence_threshold
            if self.association_mode == "two_stage_global"
            else self.confidence
        )

        requested_device = str(self._value("device"))
        if requested_device == "auto":
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = requested_device

        self.model = load_model(model_path, str(self._value("model_task")))
        self.model_is_engine = model_path.suffix.lower() == ".engine"
        self.engine_contract = None
        self.actual_inference_precision = None
        self.async_enabled = bool(self._value("scheduling.enabled"))
        self.max_detection_age = float(self._value("scheduling.max_result_age_seconds"))
        self.max_tracking_age = float(self._value("scheduling.max_track_age_seconds"))
        self.motion_max_gap = float(self._value("scheduling.motion_max_gap_seconds"))
        if not 0 < self.max_detection_age <= self.max_tracking_age or not 0 < self.motion_max_gap < 1:
            raise ValueError("Invalid async detection/tracking age or motion gap")
        self.motion = RoiMotion(int(self._value("scheduling.motion_max_width")),
                                float(self._value("scheduling.motion_max_fb_error")),
                                stationary_fallback=bool(self._value("scheduling.stationary_context_fallback")))
        self.previous_gray = None
        self.previous_sample_time = None
        self.epoch = 0
        self.discarded_detector_results = 0
        self.last_detection_stamp = None
        self.last_inference_error = None
        self.session_id = uuid4().hex
        self.roi_history = {}
        self.roi_continuity = RoiContinuity(self._parameters_with_prefix("roi_continuity."))
        self.light_learners = {}
        self.reference_bank = LightReferenceBank()
        self.periodic_targets = {}
        self.online_parameters = self._parameters_with_prefix("online_light.")
        self.periodic_parameters = {
            "min_hz": max(float(self._value("temporal.blink_min_hz")),
                          float(self._value("temporal.template_target_hz"))-float(self._value("temporal.template_frequency_tolerance"))),
            "max_hz": min(float(self._value("temporal.blink_max_hz")),
                          float(self._value("temporal.template_target_hz"))+float(self._value("temporal.template_frequency_tolerance"))),
            "min_agreement": float(self._value("temporal.photometric_min_agreement")),
            "assist_min_agreement": min(float(self._value("temporal.template_min_agreement")),
                                        float(self._value("temporal.photometric_min_agreement"))),
            "phase_steps": int(self._value("temporal.template_phase_steps")),
        }
        OnlineLightReference(self.online_parameters, self.periodic_parameters)
        self.online_generation = 0
        self.online_applied_generation = 0
        self.lamp_locator = LampRegionLocator(self._parameters_with_prefix("lamp_region."))
        self.lamp_photometer = LampPhotometer(self._parameters_with_prefix("lamp_photometry."))
        self.lamp_regions = {}
        self.lamp_measurements = {}
        self.lamp_frame_stamps = {}
        self.lamp_transforms = {}
        self.lamp_transform_stamps = {}
        self.motion_exclusions = ()
        self.independent_roi_tracking = bool(self._value("scheduling.independent_roi_tracking"))
        self.adaptive_scheduling = bool(self._value("scheduling.adaptive_enabled"))
        self.motion_speed_threshold = float(self._value("scheduling.motion_speed_threshold"))
        self.stable_hold_seconds = float(self._value("scheduling.stable_hold_seconds"))
        self.stable_motion_quality = float(self._value("scheduling.stable_motion_quality"))
        self.stable_since = None
        self.risk_scheduler = RiskScheduler(self._parameters_with_prefix("risk."))
        self.risk_status = {"enabled": self.risk_scheduler.p["enabled"]}

        if not (0 < self.motion_speed_threshold <= 10 and 0 <= self.stable_hold_seconds <= 10
                and 0 <= self.stable_motion_quality <= 1):
            raise ValueError("Invalid adaptive scheduling thresholds")
        self.tracker = MultiBeaconTracker(
            iou_threshold=float(self._value("tracking.iou_threshold")),
            center_distance_ratio=float(
                self._value("tracking.center_distance_ratio")
            ),
            max_age_seconds=float(self._value("tracking.max_age_seconds")),
            analyzer_parameters=self._parameters_with_prefix("temporal."),
            bbox_smoothing_alpha=float(
                self._value("tracking.bbox_smoothing_alpha")
            ),
            association_mode=self.association_mode,
            high_confidence_threshold=self.high_confidence_threshold,
            low_confidence_threshold=self.low_confidence_threshold,
            low_stage_iou_threshold=float(
                self._value("tracking.low_stage_iou_threshold")
            ),
            low_stage_center_distance_ratio=float(
                self._value(
                    "tracking.low_stage_center_distance_ratio"
                )
            ),
            motion_horizon_seconds=float(
                self._value("tracking.motion_horizon_seconds")
            ),
            temporal_reset_distance_ratio=float(
                self._value(
                    "tracking.temporal_reset_distance_ratio"
                )
            ),
            decision_parameters=self._parameters_with_prefix(
                "state_change."
            ),
            reference_parameters=self._parameters_with_prefix(
                "blink_reference."
            ),
            stationary_reid_enabled=bool(
                self._value("tracking.stationary_reid_enabled")
            ),
            stationary_reid_max_age_seconds=float(
                self._value("tracking.stationary_reid_max_age_seconds")
            ),
            stationary_reid_min_hits=int(
                self._value("tracking.stationary_reid_min_hits")
            ),
            stationary_reid_iou_threshold=float(
                self._value("tracking.stationary_reid_iou_threshold")
            ),
            stationary_reid_center_distance_ratio=float(
                self._value(
                    "tracking.stationary_reid_center_distance_ratio"
                )
            ),
            stationary_reid_size_similarity_min=float(
                self._value(
                    "tracking.stationary_reid_size_similarity_min"
                )
            ),
            steady_off_parameters=self._parameters_with_prefix("steady_off."),
            **{name: self._value("tracking." + name) for name in (
                "duplicate_suppression_enabled", "duplicate_iou_threshold",
                "duplicate_containment_threshold", "duplicate_center_distance_ratio",
                "weak_birth_enabled", "weak_birth_confidence", "weak_birth_min_hits",
                "weak_birth_min_seconds", "weak_birth_max_gap_seconds",
            )},
        )

        self.bridge = CvBridge()
        self.input_stream = NativeStreamStatus()
        self.last_processed_time = 0.0
        self.last_debug_time = 0.0
        self.sequence = 0

        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        debug_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            # Keep this RELIABLE for compatibility with the ROS 2 Humble
            # image_view subscriber used during Task 1 testing.
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = None
        if enable_subscription:
            self.subscription = self.create_subscription(
                Image,
                self.image_topic,
                self._image_callback,
                camera_qos,
            )
        self.debug_publisher = self.create_publisher(
            Image, self.debug_topic, debug_qos
        )
        self.result_publisher = self.create_publisher(
            String, self.result_topic, output_qos
        )
        self.input_status_publisher = self.create_publisher(
            String, "/task1/beacons/input_status_json", output_qos
        )
        self.input_status_timer = self.create_timer(1.0, self._publish_input_status)
        self.localizer = PlaneLocalizationROS(
            self, bool(self._value("position.enabled")),
            str(self._value("position.config_file")), str(self._value("position.result_topic")),
        )
        self.frame_positions = {}
        self.localization_summary = {"status": "DISABLED", "processing_ms": 0.0}
        self.frame_timestamp_source = "ros_image_header" if enable_subscription else "host_after_capture_read"

        device_name = (
            torch.cuda.get_device_name(0)
            if self.device != "cpu" and torch.cuda.is_available()
            else "CPU"
        )
        self.get_logger().info(
            f"Task 1 ready: input={self.image_topic}, model={model_path}, "
            f"device={device_name}"
        )
        self.get_logger().info(
            f"Algorithm={self.algorithm_version}, "
            f"association={self.association_mode}, "
            f"color=ring_aperture, "
            f"YOLO candidate conf={self.inference_confidence:.2f}, "
            f"new-track conf={self.high_confidence_threshold:.2f}"
        )
        self.get_logger().info(
            f"Outputs: json={self.result_topic}, debug={self.debug_topic} "
            f"({self.debug_max_fps:.1f} FPS, "
            f"{'continuous' if self.debug_always_publish else 'on demand'})"
        )

        if self.async_enabled:
            self.detector_worker = ScheduledDetector(
                self._predict_detections,
                search_fps=float(self._value("scheduling.search_fps")),
                track_fps=float(self._value("scheduling.track_fps")),
                duty_cycle=float(self._value("scheduling.duty_cycle")),
                min_idle_seconds=float(self._value("scheduling.min_idle_seconds")),
                budget_enabled=bool(self._value("scheduling.budget_enabled")),
                motion_fps=float(self._value("scheduling.motion_fps")),
                enabled=self.mission_control.active,
            )
            self.get_logger().info("Async photometry active; YOLO budget=" +
                                   ("enabled" if self._value("scheduling.budget_enabled") else "OFF"))
        self.add_on_set_parameters_callback(self._live_parameters)
        self.control_subscription = None
        if self.mission_control.enabled:
            self.control_subscription = self.create_subscription(
                String, str(self._value("mission_control.topic")), self._mission_command,
                QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                           reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def _mission_command(self, message):
        if not self.mission_control.update(message.data):
            return
        if self.detector_worker is not None:
            self.detector_worker.set_enabled(self.mission_control.active)
        self.get_logger().info(
            f"Mission perception: mode={self.mission_control.mode}, "
            f"buoy={'ON' if self.mission_control.active else 'OFF'}")

    def _live_parameters(self, changes):
        values = {parameter.name: parameter.value for parameter in changes}
        online = dict(self.online_parameters)
        zoom = dict(self.zoom_parameters)
        continuity = dict(self.roi_continuity.p)
        control = {key: self._value("scheduling." + key) for key in CONTROL_DEFAULTS}
        tuning = {"scheduling.motion_speed_threshold": self.motion_speed_threshold,
                  "scheduling.stable_hold_seconds": self.stable_hold_seconds,
                  "scheduling.stable_motion_quality": self.stable_motion_quality}
        try:
            for name, value in values.items():
                if name.startswith("online_light.") and name[13:] in ONLINE_LIGHT_DEFAULTS:
                    online[name[13:]] = value
                elif name.startswith("local_zoom.") and name[11:] in LOCAL_ZOOM_DEFAULTS:
                    zoom[name[11:]] = value
                elif name.startswith("roi_continuity.") and name[15:] in ROI_CONTINUITY_DEFAULTS:
                    continuity[name[15:]] = value
                elif name.startswith("scheduling.") and name[11:] in CONTROL_DEFAULTS:
                    control[name[11:]] = value
                elif name in tuning:
                    tuning[name] = float(value)
                elif name in self._parameter_names:
                    raise ValueError(name + " requires editing detector configuration and restarting")
            OnlineLightReference(online, self.periodic_parameters)
            zoom = validate_zoom(zoom)
            continuity_instance = RoiContinuity(continuity)
            validate_control(control)
            if not 0 < tuning["scheduling.motion_speed_threshold"] <= 10:
                raise ValueError("motion_speed_threshold must be in (0, 10]")
            if not 0 <= tuning["scheduling.stable_hold_seconds"] <= 10:
                raise ValueError("stable_hold_seconds must be in [0, 10]")
            if not 0 <= tuning["scheduling.stable_motion_quality"] <= 1:
                raise ValueError("stable_motion_quality must be in [0, 1]")
        except (ValueError, TypeError) as error:
            return SetParametersResult(successful=False, reason=str(error))
        if online != self.online_parameters:
            self.online_parameters = online
            self.online_generation += 1
        self.zoom_parameters = zoom
        if continuity != self.roi_continuity.p:
            # Replace as one object so the capture thread sees a coherent policy.
            # Relearn only the spatial anchor, not an already measured blink.
            self.roi_continuity = continuity_instance
        if self.detector_worker:
            self.detector_worker.configure(control)
        self.motion_speed_threshold = tuning["scheduling.motion_speed_threshold"]
        self.stable_hold_seconds = tuning["scheduling.stable_hold_seconds"]
        self.stable_motion_quality = tuning["scheduling.stable_motion_quality"]
        return SetParametersResult(successful=True)

    def _declare_parameters(self) -> None:
        defaults = {
            "mission_control.enabled": False,
            "mission_control.topic": "/tongji_mrl/tsky/downward_vision_control",
            "model_path": "",
            "image_topic": "/arducam/image_raw",
            "model_task": "segment",
            "debug_topic": "/task1/beacons/debug_image",
            "debug_max_fps": 10.0,
            "debug_always_publish": True,
            "debug_show_missing_tracks": False,
            "debug_image_size": [960, 600],
            "result_topic": "/task1/beacons/results_json",
            "device": "auto",
            "algorithm_version": "NEW_BEACON_RING_APERTURE",
            "confidence": 0.40,
            "image_size": 640,
            "processing.cpu_threads": 2,
            "inference_precision": 32,
            "max_detections": 10,
            # Process every latest-frame QoS sample. A limiter exactly equal
            # to the nominal camera FPS drops alternating frames under jitter.
            "max_fps": 0.0,
            # Include a narrow outer ring used as local background evidence.
            "roi_padding_ratio": 0.12,
            "minimum_confirmed_hits": 2,
            "tracking.iou_threshold": 0.20,
            "tracking.center_distance_ratio": 1.25,
            "tracking.bbox_smoothing_alpha": 0.70,
            # Bridge the official one-second dark phase and USB frame jitter.
            "tracking.max_age_seconds": 1.8,
            "tracking.association_mode": "greedy",
            "tracking.high_confidence_threshold": 0.40,
            "tracking.low_confidence_threshold": 0.20,
            "tracking.low_stage_iou_threshold": 0.10,
            "tracking.low_stage_center_distance_ratio": 0.75,
            "tracking.motion_horizon_seconds": 0.35,
            "tracking.temporal_reset_distance_ratio": 0.0,
            "tracking.color_roi_bbox_source": "smoothed",
            # Keep expired boxes briefly for strict same-place recovery.
            "tracking.stationary_reid_enabled": False,
            "tracking.stationary_reid_max_age_seconds": 6.0,
            "tracking.stationary_reid_min_hits": 3,
            "tracking.stationary_reid_iou_threshold": 0.10,
            "tracking.stationary_reid_center_distance_ratio": 0.40,
            "tracking.stationary_reid_size_similarity_min": 0.60,
            "tracking.duplicate_suppression_enabled": False,
            "tracking.duplicate_iou_threshold": 0.45,
            "tracking.duplicate_containment_threshold": 0.85,
            "tracking.duplicate_center_distance_ratio": 0.30,
            "tracking.weak_birth_enabled": False,
            "tracking.weak_birth_confidence": 0.18,
            "tracking.weak_birth_min_hits": 4,
            "tracking.weak_birth_min_seconds": 0.60,
            "tracking.weak_birth_max_gap_seconds": 0.65,
            "steady_off.enabled": False,
            "steady_off.seconds": 4.0,
            "steady_off.min_samples": 16,
            "steady_off.max_gap_seconds": 0.65,
            "steady_off.min_color_confidence": 0.55,
            "steady_off.absolute_signal_range": 0.025,
            "steady_off.relative_signal_range": 0.12,
            "steady_off.max_neutral_clip_ratio": 0.50,
            "steady_off.blink_cooldown_seconds": 2.2,
            "steady_off.max_center_motion_ratio": 0.25,
            "temporal_color_evidence_mode": "legacy_stabilized",
            # Learn a persistent dark/bright range only from a trusted blink.
            "blink_reference.enabled": False,
            "blink_reference.min_samples": 12,
            "blink_reference.min_agreement": 0.82,
            "blink_reference.min_signal_separation": 0.045,
            "blink_reference.low_percentile": 20.0,
            "blink_reference.high_percentile": 82.0,
            "blink_reference.ema_alpha": 0.20,
            "blink_reference.dark_enter_ratio": 0.38,
            "blink_reference.lit_exit_ratio": 0.62,
            "color.mode": "hsv",
            "color.saturation_min": 90,
            "color.value_min": 120,
            "color.value_percentile": 80.0,
            "color.min_color_ratio": 0.0025,
            "color.min_color_margin": 0.00075,
            "color.background_border_ratio": 0.10,
            "color.background_subtract_weight": 1.0,
            "color.min_component_ratio": 0.0008,
            "color.min_brightness_contrast": 12.0,
            "color.lit_confirm_frames": 2,
            "color.dark_confirm_frames": 2,
            "color.color_switch_frames": 3,
            "color.off_without_lit_min_hits": 15,
            "color.off_without_lit_min_detector_confidence": 0.45,
            "color.red_low_max": 10,
            "color.red_high_min": 170,
            "color.green_min": 35,
            "color.green_max": 90,
            "color.blue_min": 95,
            "color.blue_max": 135,
            "color.channel_dominance_min": 0.03,
            "color.chromaticity_spread_min": 0.06,
            "color.center_region_ratio": 0.56,
            "color.center_saturation_percentile": 75.0,
            "color.center_value_percentile": 90.0,
            "color.min_center_saturation": 80.0,
            "color.min_emission_strength": 20.0,
            "color.min_center_color_fraction": 0.10,
            "color.strict_hue_share_min": 0.80,
            "color.strict_hue_margin_min": 0.65,
            "color.strict_channel_share_min": 0.45,
            "color.strict_channel_margin_min": 0.08,
            "color.color_brightness_percentile": 0.0,
            "color.radial_weight_strength": 0.0,
            "color.blue_min_center_saturation": 0.0,
            "color.blue_min_emission_strength": 0.0,
            "color.blue_strict_hue_share_min": 0.80,
            "color.blue_strict_channel_share_min": 0.45,
            "color.max_neutral_clip_ratio_for_blue": 1.0,
            "color.weak_blue_as_dark": False,
            "color.emitter_search_ratio": 0.76,
            "color.emitter_core_radius_ratio": 0.075,
            "color.emitter_hint_radius_ratio": 0.18,
            "color.emitter_hint_update_confidence_min": 0.55,
            "color.emitter_hint_update_alpha": 0.20,
            "color.emitter_ring_inner_ratio": 0.055,
            "color.emitter_ring_outer_ratio": 0.26,
            "color.emitter_color_percentile": 58.0,
            "color.emitter_min_signal": 0.055,
            "color.emitter_red_signal_min": 0.055,
            "color.emitter_green_signal_min": 0.055,
            "color.emitter_color_share_min": -1.0,
            "color.emitter_color_margin_min": -1.0,
            "color.emitter_min_chroma_fraction": 0.012,
            "color.emitter_hue_share_min": 0.62,
            "color.emitter_hue_margin_min": 0.24,
            "color.emitter_channel_share_min": 0.43,
            "color.emitter_channel_margin_min": 0.06,
            "color.emitter_blue_share_min": 0.78,
            "color.emitter_blue_signal_min": 0.095,
            "color.emitter_blue_chroma_min": 0.18,
            "color.emitter_blue_core_contrast_min": 12.0,
            "color.emitter_fragmented_signal_min": 0.20,
            "color.emitter_blue_core_bypass_signal": 0.30,
            "temporal.history_seconds": 5.0,
            "temporal.blink_min_window": 2.8,
            "temporal.blink_min_transitions": 3,
            "temporal.blink_min_hz": 0.35,
            "temporal.blink_max_hz": 0.65,
            "temporal.solid_min_seconds": 1.5,
            # A 0.5 Hz blink has a nominal one-second OFF phase. Keep this
            # longer so a valid blink never briefly becomes OFF.
            "temporal.off_min_seconds": 1.6,
            "temporal.color_purity_min": 0.70,
            "temporal.mode": "transitions",
            "temporal.template_target_hz": 0.50,
            "temporal.template_frequency_tolerance": 0.10,
            "temporal.template_min_agreement": 0.78,
            "temporal.template_phase_steps": 40,
            "temporal.template_missing_weight": 0.55,
            "temporal.photometric_signal_delta_min": 0.025,
            "temporal.photometric_low_percentile": 20.0,
            "temporal.photometric_high_percentile": 82.0,
            "temporal.photometric_low_threshold": 0.32,
            "temporal.photometric_high_threshold": 0.68,
            "temporal.photometric_min_samples": 12,
            "temporal.photometric_min_agreement": 0.72,
            "temporal.photometric_color_purity_min": 0.55,
            "temporal.photometric_blue_purity_min": 0.0,
            "temporal.photometric_color_level_min": 0.015,
            "temporal.photometric_blue_level_min": 0.18,
            "temporal.photometric_color_delta_min": 0.008,
            "temporal.photometric_color_phase_mode": "independent",
            "state_change.enabled": False,
            "state_change.candidate_min_frames": 2,
            "state_change.candidate_min_seconds": 0.12,
            "state_change.candidate_hold_seconds": 1.20,
            "state_change.commit_min_frames": 2,
            "state_change.commit_confidence_min": 0.55,
            "state_change.change_pending_seconds": 0.30,
            "state_change.change_clear_seconds": 0.40,
            "state_change.blink_phase_timeout_seconds": 1.30,
        }
        defaults.update({
            "scheduling.enabled": False,
            "scheduling.search_fps": 4.0,
            "scheduling.track_fps": 3.0,
            "scheduling.duty_cycle": 0.55,
            "scheduling.min_idle_seconds": 0.04,
            "scheduling.max_result_age_seconds": 0.85,
            "scheduling.max_track_age_seconds": 1.50,
            "scheduling.motion_max_gap_seconds": 0.35,
            "scheduling.motion_max_width": 1280,
            "scheduling.motion_max_fb_error": 1.5,
            "scheduling.stationary_context_fallback": False,
            "scheduling.budget_enabled": True,
            "scheduling.motion_fps": 6.0,
            "scheduling.adaptive_enabled": False,
            "scheduling.independent_roi_tracking": False,
            "scheduling.motion_speed_threshold": 0.25,
            "scheduling.stable_hold_seconds": 1.0,
            "scheduling.stable_motion_quality": 0.75,
            "temporal.photometric_max_gap_seconds": 0.35,
        })
        defaults.update({"color." + key: value for key, value in APERTURE_DEFAULTS.items()})
        defaults.update({"color." + key: value for key, value in EMITTER_FUSION_DEFAULTS.items()})
        defaults.update({"online_light." + key: value for key, value in ONLINE_LIGHT_DEFAULTS.items()})
        defaults.update({"roi_continuity." + key: value for key, value in ROI_CONTINUITY_DEFAULTS.items()})
        defaults.update({"local_zoom." + key: value for key, value in LOCAL_ZOOM_DEFAULTS.items()})
        defaults.update({"risk." + key: value for key, value in RISK_DEFAULTS.items()})
        defaults.update({"lamp_region." + key: value for key, value in LAMP_REGION_DEFAULTS.items()})
        defaults.update({"lamp_photometry." + key: value for key, value in LAMP_PHOTOMETRY_DEFAULTS.items()})
        defaults.update({"position.enabled": False, "position.config_file": "",
                         "position.result_topic": "/task1/beacons/positions_json"})
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self._parameter_names = tuple(defaults)

    def _value(self, name: str):
        return self.get_parameter(name).value

    def _parameters_with_prefix(self, prefix: str) -> Dict[str, object]:
        return {
            name[len(prefix):]: self._value(name)
            for name in self._parameter_names
            if name.startswith(prefix)
        }

    def _image_callback(self, message: Image) -> None:
        # Count native arrivals before conversion or the CPU processing cap.
        self.input_stream.received(message.width, message.height, message.encoding, message.header)
        if not self.mission_control.active:
            return
        if self.max_fps > 0 and time.monotonic() - self.last_processed_time < 1.0 / self.max_fps:
            return  # Avoid converting a ROS image that the CPU rate cap will discard.
        try:
            frame = self.bridge.imgmsg_to_cv2(message, "bgr8")
        except Exception as error:
            self.get_logger().error(f"Image conversion failed: {error}")
            return
        self.process_frame(frame, message.header)

    def process_frame(self, frame, header) -> None:
        """Process a BGR frame received from the ROS image subscription."""
        if not self.mission_control.active:
            return False
        wall_now = time.monotonic()
        if self.max_fps > 0.0:
            interval = 1.0 / self.max_fps
            if wall_now - self.last_processed_time < interval:
                return False
            # Accumulate the schedule: resetting to now would turn a 12 FPS
            # cap on a 15 FPS camera into 7.5 FPS by discarding every other frame.
            self.last_processed_time = (self.last_processed_time + interval
                if wall_now - self.last_processed_time < 2 * interval else wall_now)
        else:
            self.last_processed_time = wall_now
        sample_time = self._header_time(header)

        started = time.perf_counter()
        if self.async_enabled:
            tracks = self._async_tracks(frame, sample_time)
        else:
            try:
                detections = self._predict_detections(frame)
            except Exception as error:
                self.get_logger().error(f"YOLO inference failed: {error}")
                return False
            tracks = self.tracker.update(detections, sample_time)
        tracked_at = time.perf_counter()
        online_parameters = dict(self.online_parameters)
        if self.online_applied_generation != self.online_generation:
            self._reset_photometry_evidence(sample_time)
            self.online_applied_generation = self.online_generation
        active_keys = {(track.track_id, track.roi_epoch) for track in tracks}
        self.lamp_locator.prune(active_keys)
        self.lamp_regions = {key: value for key, value in self.lamp_regions.items() if key in active_keys}
        self.lamp_frame_stamps = {key: value for key, value in self.lamp_frame_stamps.items() if key in active_keys}
        self.light_learners = {key: learner for key, learner in self.light_learners.items() if key in active_keys}
        self.roi_continuity.prune(active_keys, sample_time)
        self.reference_bank.prune(active_keys, sample_time)
        # Validate every current lamp before any target can borrow a reference.
        self.lamp_measurements = {}
        for track in tracks:
            if not track.visible or (self.risk_scheduler.p["enabled"] and track.hits < self.minimum_confirmed_hits):
                continue
            key = (track.track_id, track.roi_epoch)
            box = analysis_bbox(track, self.color_roi_bbox_source)
            previous = self.lamp_regions.get(key)
            matrix = self.lamp_transforms.get(track.track_id)
            if (previous is None or not previous.valid or
                    self.lamp_frame_stamps.get(key) != self.lamp_transform_stamps.get(track.track_id)):
                matrix = None
            region = self.lamp_locator.update(key, frame, box, sample_time, transform=matrix)
            self.lamp_regions[key] = region
            self.lamp_frame_stamps[key] = sample_time
            if region.reset:
                self._reset_track_photometry(track, sample_time, "lamp_geometry_changed")
            self.lamp_measurements[key] = self.lamp_photometer.measure(
                frame, region, self._padded_bbox(frame.shape, box))
        self._update_periodic_targets(
            [track for track in tracks if track.hits >= self.minimum_confirmed_hits], sample_time)
        for track in tracks:
            if self.risk_scheduler.p["enabled"] and track.hits < self.minimum_confirmed_hits:
                # Detector confirmations establish identity; optical flow and
                # two brightness clusters cannot promote a tentative candidate.
                observation = HsvColorClassifier._empty()
                observation.decision_reason = "TENTATIVE_IDENTITY"
                observation.photometric_phase = "UNOBSERVED"
                track.raw_instant = track.instant = observation
                continue
            if not track.visible:
                # Missing is visibility evidence, never a direct OFF vote.
                update_missing_track(track, sample_time)
                continue
            key = (track.track_id, track.roi_epoch)
            observation = self.lamp_measurements[key]
            if online_parameters["enabled"]:
                key = (track.track_id, track.roi_epoch)
                if key not in self.light_learners:
                    self.light_learners[key] = OnlineLightReference(online_parameters, self.periodic_parameters)
                observation = self.light_learners[key].update(
                    observation, sample_time, analysis_bbox(track, self.color_roi_bbox_source),
                    track.motion_quality if self.async_enabled else 1.0,
                    allow_learning=(not self.risk_scheduler.p["enabled"]
                                    or track.association_stage != "CONTEXT_RECOVERED"),
                    peer_references=self.reference_bank.peers(key),
                )
                if observation.emission_diagnostics.get("online_light", {}).get("history_reset"):
                    self.reference_bank.entries.pop(key, None)
                    track.analyzer.reset()
                    track.confirmed_instant_label = "UNKNOWN"
                    track.pending_instant_label, track.pending_instant_count = "UNKNOWN", 0
            if self.async_enabled and online_parameters["enabled"]:
                self.roi_continuity.observe(
                    (track.track_id, track.roi_epoch), sample_time,
                    analysis_bbox(track, self.color_roi_bbox_source), self.previous_gray,
                    observation, track.motion_quality, float(online_parameters["min_motion_quality"]),
                    allow_anchor_refresh=track.association_stage != "CONTEXT_RECOVERED",
                )
            update_track_observation(
                track,
                observation,
                sample_time,
                lit_confirm_frames=self.lit_confirm_frames,
                dark_confirm_frames=self.dark_confirm_frames,
                color_switch_frames=self.color_switch_frames,
                off_without_lit_min_hits=self.off_without_lit_min_hits,
                off_without_lit_min_detector_confidence=(
                    self.off_without_lit_min_detector_confidence
                ),
                emitter_hint_update_confidence_min=(
                    self.emitter_hint_update_confidence_min
                ),
                emitter_hint_update_alpha=self.emitter_hint_update_alpha,
                color_evidence_mode=self.temporal_color_evidence_mode,
                guard_online_dark=self.risk_scheduler.p["enabled"],
            )

        # Motion sees only stable housing/context, excluding the actual lamp head.
        exclusions = []
        for region in self.lamp_regions.values():
            if region.valid and region.outer is not None:
                (cx, cy), axes, _ = region.outer
                radius = max(axes) * 0.60
                exclusions.append((cx-radius, cy-radius, cx+radius, cy+radius))
        self.motion_exclusions = tuple(exclusions)
        color_finished = time.perf_counter()
        self.input_stream.processed()
        confirmed = [
            track for track in tracks
            if track.hits >= self.minimum_confirmed_hits
        ]
        self._update_periodic_targets(confirmed, sample_time)
        if self.localizer.enabled:
            observations = []
            for track in confirmed:
                region = self.lamp_regions.get((track.track_id, track.roi_epoch))
                anchor = (region.outer[0] if region is not None and region.valid
                    and region.quality >= self.online_parameters["min_measurement_quality"] else None)
                observations.append({"track_id": track.track_id, "visible": track.visible,
                    "bbox": analysis_bbox(track, self.color_roi_bbox_source),
                    "motion_quality": track.motion_quality if self.async_enabled else 1.0,
                    "pixel_uv": anchor, "pixel_source": "LAMP_PLATE_CENTER"})
            self.frame_positions, self.localization_summary = self.localizer.locate(observations, frame.shape, header)
        processing_ms = (time.perf_counter() - started) * 1000.0
        self.sequence += 1
        payload = {
            "schema_version": 4 if self.localizer.enabled else (3 if self.async_enabled else (2 if self.state_change_enabled else 1)),
            "session_id": self.session_id,
            "algorithm_version": self.algorithm_version,
            "state_model": (
                "candidate_committed"
                if self.state_change_enabled else "raw_temporal"
            ),
            "sequence": self.sequence,
            "stamp": {
                "sec": header.stamp.sec,
                "nanosec": header.stamp.nanosec,
            },
            "frame_id": header.frame_id,
            "image_size": [int(frame.shape[1]), int(frame.shape[0])],
            "timestamp_source": self.frame_timestamp_source,
            "model_task": self.model.task,
            "box_convention": "MASK_AABB" if self.model.task == "segment" else "DETECTOR_AABB",
            "processing_ms": round(processing_ms, 2),
            "roi_ms": round((tracked_at - started) * 1000.0, 2),
            "color_temporal_ms": round((color_finished - tracked_at) * 1000.0, 2),
            "inference_region": dict(self.last_inference_region),
            "photometry_device": "cpu_native_roi",
            "input_stream": self.input_stream.snapshot(),
            "inference_precision": self.actual_inference_precision,
            "detector_runtime": self.detector_worker.snapshot() if self.detector_worker else {"schedule": "SYNCHRONOUS"},
            "detection_age_seconds": (round(sample_time - self.last_detection_stamp, 3)
                                      if self.last_detection_stamp is not None else None),
            "discarded_detector_results": self.discarded_detector_results,
            "scheduling_risk": dict(self.risk_status),
            "suppressed_detections": self.tracker.suppressed_detections,
            "periodic_reference_bank": self.reference_bank.snapshot(),
            "priority_beacon_count": sum(item["priority"] for item in self.periodic_targets.values()),
            "beacon_count": sum(track.visible for track in confirmed),
            "tracks": [
                self._track_payload(track, sample_time)
                for track in confirmed
            ],
        }
        if self.localizer.enabled:
            payload["image_size"] = [int(frame.shape[1]), int(frame.shape[0])]
            payload["timestamp_source"] = self.frame_timestamp_source
            payload["localization"] = self.localization_summary
            for track_payload in payload["tracks"]:
                track_payload["position"] = self.frame_positions[track_payload["track_id"]]
        output = String()
        self.last_payload = payload
        output.data = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        )
        if not self._safe_publish(self.result_publisher, output):
            return
        if self.localizer.enabled:
            self.localizer.publish(payload)
        self._publish_debug(header, frame, confirmed, processing_ms)
        return True

    def _predict_detections(self, frame, context=None):
        self.last_inference_region = {**self.last_inference_region, "in_progress": True,
                                      "backend": "TensorRT" if self.model_is_engine else "PyTorch"}
        fp16 = self.device != "cpu" and self.inference_precision == 16
        precision = ({"quantize": 16 if fp16 else 32} if "quantize" in DEFAULT_CFG_DICT
                     else {"half": fp16})
        if self.model_is_engine and self.engine_contract is None:
            try:
                self.engine_contract = engine_contract(self.model, self.device, precision, self.image_size)
            except Exception as error:
                self.last_inference_region = {"kind": "ERROR", "reason": str(error), "backend": "TensorRT"}
                raise
            self.get_logger().info("TensorRT verified input " + str(self.engine_contract["shape_bchw"]) +
                                   "; full frames and crops share this shape")
        context = context if context is not None else {"force_full": True, "epoch": self.epoch}
        zoom = dict(context.get("zoom", self.zoom_parameters))
        region, cropped = self.region_planner.choose(
            frame.shape, context, zoom, context.get("region_time_seconds", time.monotonic()),
            fixed_engine=self.model_is_engine)
        left, top, right, bottom = region
        source = frame[top:bottom, left:right]
        size = (tuple(self.engine_contract["shape_bchw"][2:]) if self.model_is_engine else
                int(zoom["image_size"]) if cropped else self.image_size)
        context["region_used"] = tuple(region)
        context["cropped"] = cropped
        # One region and one forward pass per scheduled slot, never full+crop.
        kwargs = dict(source=source, conf=self.inference_confidence,
                      imgsz=size, device=self.device, classes=[0],
                      max_det=self.max_detections, verbose=False,
                      rect=not self.model_is_engine)
        # 8.4 uses quantize; earlier releases use half. Do not retry an entire
        # failed inference or silently switch to full precision on the GPU.
        kwargs.update(precision)
        try:
            with torch.inference_mode():
                prediction = self.model.predict(**kwargs)[0]
                detections = detections_from_prediction(prediction, source.shape,
                                                       require_masks=self.model.task == "segment")
        except Exception as error:
            self.last_inference_region = {"kind": "ERROR", "in_progress": False,
                "reason": str(error), "engine_contract": self.engine_contract,
                "backend": "TensorRT" if self.model_is_engine else "PyTorch"}
            raise
        if cropped:
            # Partial boxes at an artificial crop edge cannot redefine a lamp ROI.
            detections = [Detection((d.bbox[0] + left, d.bbox[1] + top,
                                     d.bbox[2] + left, d.bbox[3] + top), d.confidence, d.class_id)
                          for d in detections
                          if (left == 0 or d.bbox[0] > 2)
                          and (top == 0 or d.bbox[1] > 2)
                          and (right == frame.shape[1] or d.bbox[2] < source.shape[1] - 2)
                          and (bottom == frame.shape[0] or d.bbox[3] < source.shape[0] - 2)]
        self.last_inference_region = {"kind": "LOCAL" if cropped else "FULL",
            "in_progress": False,
            "bbox_xyxy": list(region), "image_size": size,
            "reason": self.region_planner.last_reason, "zoom_requested": bool(zoom["enabled"]),
            # For .pt rect=True can use less padding than predictor.imgsz.
            # Report the verified engine tensor size, not a guessed .pt shape.
            "input_shape_hw": list(self.engine_contract["shape_bchw"][2:]) if self.model_is_engine else None,
            "letterbox_target_hw": list(self.model.predictor.imgsz),
            "backend": "TensorRT" if self.model_is_engine else "PyTorch",
            "engine_contract": self.engine_contract,
            "zoom_size_policy": "ENGINE_SHAPE" if self.model_is_engine else "PROFILE_SIZE",
            "coordinates": "NATIVE_FRAME", "resize": "LETTERBOX"}
        backend = getattr(self.model.predictor, "model", None)
        self.actual_inference_precision = 16 if getattr(backend, "fp16", False) else 32
        # This is the worker's actual successful completion, not a reused packet.
        self.input_stream.yolo_completed()
        return detections

    def _publish_input_status(self):
        status = self.input_stream.snapshot()
        output = String()
        output.data = json.dumps(status, ensure_ascii=False, separators=(",", ":"))
        self._safe_publish(self.input_status_publisher, output)
        self.get_logger().info(status_text(status))

    def _update_periodic_targets(self, tracks, stamp):
        self.periodic_targets = {}
        enabled = self.online_parameters["enabled"] and self.online_parameters["require_periodic"]
        if not enabled:
            self.reference_bank.clear()
            return
        for track in tracks:
            key = (track.track_id, track.roi_epoch)
            learner = self.light_learners.get(key)
            if learner is None:
                self.reference_bank.entries.pop(key, None)
                continue
            eligible = (track.visible
                        and key in self.lamp_measurements
                        and self.lamp_measurements[key].photometric_phase != "UNOBSERVED"
                        and self.lamp_measurements[key].measurement_quality >= self.online_parameters["min_measurement_quality"]
                        and track.association_stage != "CONTEXT_RECOVERED"
                        and (not self.async_enabled or track.motion_quality >= self.online_parameters["min_motion_quality"]))
            valid = self.reference_bank.record(key, learner, stamp, eligible)
            row = self.reference_bank.entries.get(key, {})
            self.periodic_targets[track.track_id] = {
                "priority": valid, "periodic_reference_valid": valid,
                "status": learner.periodic_evidence["status"] if valid else "WAITING_FOR_OWN_PERIODIC_EVIDENCE",
                "reference_id": row.get("reference_id"), "independent": row.get("independent", False),
                "frequency_hz": row.get("frequency_hz"),
                "peer_support": learner.periodic_evidence.get("peer_support", {"support_count": 0}),
                "basis": "DETECTOR_CONFIRMED_AND_OWN_PERIODIC_LIGHT",
            }

    def _reset_photometry_evidence(self, stamp, reason="photometry_parameters_changed"):
        self.light_learners.clear()
        self.reference_bank.clear()
        self.periodic_targets.clear()
        self.roi_continuity.clear()
        self.lamp_locator.clear()
        self.lamp_regions.clear()
        self.lamp_measurements.clear()
        self.lamp_frame_stamps.clear()
        self.lamp_transforms.clear()
        self.lamp_transform_stamps.clear()
        self.motion_exclusions = ()
        # Dormant IDs may return later; their old photometric references are invalid too.
        for track in list(self.tracker.tracks.values()) + list(self.tracker.dormant_tracks.values()):
            track.analyzer.reset()
            track.blink_reference.reset()
            track.steady_off.reset()
            track.decision.reset(timestamp=stamp, identity_reset=False)
            track.decision.source = reason
            track.temporal = TemporalResult()
            track.raw_instant = track.instant = HsvColorClassifier._empty()
            track.confirmed_instant_label = track.pending_instant_label = "UNKNOWN"
            track.pending_instant_count = 0
            track.last_reliable_color = "UNKNOWN"

    def _reset_track_photometry(self, track, stamp, reason):
        key = (track.track_id, track.roi_epoch)
        self.light_learners.pop(key, None)
        self.reference_bank.entries.pop(key, None)
        self.periodic_targets.pop(track.track_id, None)
        for mapping in (self.roi_continuity.anchors, self.roi_continuity.references,
                        self.roi_continuity.pending, self.roi_continuity.last_status):
            mapping.pop(key, None)
        track.analyzer.reset()
        track.blink_reference.reset()
        track.steady_off.reset()
        track.decision.reset(timestamp=stamp, identity_reset=False)
        track.decision.source = reason
        track.temporal = TemporalResult()
        track.confirmed_instant_label = track.pending_instant_label = "UNKNOWN"
        track.pending_instant_count = 0
        track.last_reliable_color = "UNKNOWN"
        track.emitter_center_hint = None

    def _async_tracks(self, frame, stamp):
        self.lamp_transforms = {}
        self.lamp_transform_stamps = {}
        gray = self.motion.gray(frame)
        if self.previous_sample_time is not None and stamp <= self.previous_sample_time:
            self.epoch += 1
            self.tracker.tracks.clear()
            self.tracker.dormant_tracks.clear()
            self.tracker._weak_candidates.clear()
            self.previous_gray = None
            self.last_detection_stamp = None
            self.roi_history.clear()
            self.light_learners.clear()
            self.reference_bank.clear()
            self.periodic_targets.clear()
            self.roi_continuity.clear()
            self.lamp_locator.clear()
            self.lamp_regions.clear()
            self.lamp_measurements.clear()
            self.lamp_frame_stamps.clear()
            self.motion_exclusions = ()
            self.stable_since = None
            self.risk_scheduler.reset()
        previous_positions = {key: (entry[2], entry[3]) for key, entry in self.roi_history.items()}
        transported = {}
        recovered_ids = set()
        if self.independent_roi_tracking:
            groups = {}
            for track in self.tracker.tracks.values():
                entry = self.roi_history.get(track.track_id)
                if entry is None or entry[0] != track.roi_epoch:
                    continue
                roi_epoch, reference, reference_stamp, box = entry
                if 0 < stamp - reference_stamp <= self.motion_max_gap:
                    group = groups.setdefault((reference_stamp, id(reference)), [reference, {}])
                    group[1][track.track_id] = box
            for (reference_stamp, _), (reference, boxes) in groups.items():
                transported.update(self.motion.transport(boxes, reference, gray, frame.shape,
                                                         self.motion_exclusions))
                self.lamp_transforms.update(self.motion.last_transforms)
                self.lamp_transform_stamps.update({key: reference_stamp for key in self.motion.last_transforms})
        elif self.previous_sample_time is not None and 0 < stamp - self.previous_sample_time <= self.motion_max_gap:
            boxes = {t.track_id: analysis_bbox(t, self.color_roi_bbox_source)
                     for t in self.tracker.tracks.values() if t.visible}
            transported = self.motion.transport(boxes, self.previous_gray, gray, frame.shape,
                                                self.motion_exclusions)
            self.lamp_transforms.update(self.motion.last_transforms)
            self.lamp_transform_stamps.update({key: self.previous_sample_time for key in self.motion.last_transforms})

        if self.independent_roi_tracking and self.online_parameters["enabled"]:
            required_quality = float(self.online_parameters["min_motion_quality"])
            for track in self.tracker.tracks.values():
                if transported.get(track.track_id, (None, 0.0))[1] >= required_quality:
                    continue
                recovered = self.roi_continuity.recover(
                    (track.track_id, track.roi_epoch), stamp, gray, frame.shape,
                    min_quality=required_quality,
                )
                if recovered is not None and recovered[1] >= required_quality:
                    transported[track.track_id] = recovered
                    recovered_ids.add(track.track_id)
            transported = {key: value for key, value in transported.items() if value[1] >= required_quality}

        def accept_roi(track, box, quality, stage):
            if (self.independent_roi_tracking and self.online_parameters["enabled"]
                    and quality < float(self.online_parameters["min_motion_quality"])):
                track.visible = False
                track.measurement_bbox = None
                return
            track.measurement_bbox = box
            track.measurement_time = stamp
            track.motion_quality = quality
            track.bbox = box
            track.visible = True
            track.association_stage = stage
            track.roi_tracking_active = (self.independent_roi_tracking and
                (not self.risk_scheduler.p["enabled"] or track.hits >= self.minimum_confirmed_hits))
            if track.roi_tracking_active:
                track.roi_last_observed = stamp

        for track in self.tracker.tracks.values():
            track.visible = False
            track.measurement_bbox = None
            track.roi_tracking_active = (self.independent_roi_tracking and
                (not self.risk_scheduler.p["enabled"] or track.hits >= self.minimum_confirmed_hits))
            if track.track_id in transported and (self.independent_roi_tracking or
                    stamp - track.detector_stamp <= self.max_tracking_age):
                box, quality = transported[track.track_id]
                accept_roi(track, box, quality, "CONTEXT_RECOVERED" if track.track_id in recovered_ids else "FLOW")
        packet = self.detector_worker.poll()
        if packet is not None:
            age = stamp - packet["stamp"]
            if packet["error"]:
                if packet["error"] != self.last_inference_error:
                    self.get_logger().error("YOLO worker: " + packet["error"])
                self.last_inference_error = packet["error"]
            elif packet["epoch"] != self.epoch or not 0 <= age <= self.max_detection_age:
                self.discarded_detector_results += 1
            else:
                self.last_inference_error = None
                moved = self.motion.transport(
                    {i: det.bbox for i, det in enumerate(packet["detections"])},
                    packet["gray"], gray, frame.shape,
                    self.motion_exclusions,
                )
                if self.independent_roi_tracking and self.online_parameters["enabled"]:
                    moved = {key: value for key, value in moved.items()
                             if value[1] >= float(self.online_parameters["min_motion_quality"])}
                detections = [Detection(moved[i][0], det.confidence, det.class_id)
                              for i, det in enumerate(packet["detections"]) if i in moved]
                self.last_detection_stamp = packet["stamp"]
                moved_quality = {tuple(box): quality for box, quality in moved.values()}
                cropped = bool((packet.get("context") or {}).get("cropped", False))
                tracks = self.tracker.update(detections, stamp, update_weak_candidates=not cropped)
                for track in tracks:
                    if track.visible:
                        track.detector_stamp = packet["stamp"]
                        accept_roi(track, track.detector_bbox,
                                   moved_quality.get(tuple(track.detector_bbox), 0.65), track.association_stage)
                    elif track.track_id in transported and (self.independent_roi_tracking or
                            stamp - track.detector_stamp <= self.max_tracking_age):
                        box, quality = transported[track.track_id]
                        accept_roi(track, box, quality, "CONTEXT_RECOVERED" if track.track_id in recovered_ids else "FLOW")
                self.discarded_detector_results += len(packet["detections"]) - len(moved)
        self.tracker._retire_stale_tracks(stamp)
        tracks = sorted(self.tracker.tracks.values(), key=lambda t: t.track_id)
        ready = bool(tracks) and all(t.visible and t.hits >= self.minimum_confirmed_hits for t in tracks)
        mode = "TRACK" if ready else "SEARCH"
        if self.adaptive_scheduling and not self.risk_scheduler.p["enabled"]:
            if not ready:
                self.stable_since = None
            else:
                calm = True
                for track in tracks:
                    previous = previous_positions.get(track.track_id)
                    speed = float("inf")
                    if previous is not None and stamp > previous[0]:
                        box, old = analysis_bbox(track, self.color_roi_bbox_source), previous[1]
                        movement = math.hypot((box[0] + box[2] - old[0] - old[2]) * 0.5,
                                              (box[1] + box[3] - old[1] - old[3]) * 0.5)
                        speed = movement / max(min(old[2] - old[0], old[3] - old[1]), 1) / (stamp - previous[0])
                    calm &= track.motion_quality >= self.stable_motion_quality and speed <= self.motion_speed_threshold
                if calm:
                    self.stable_since = stamp if self.stable_since is None else self.stable_since
                    mode = "TRACK" if stamp - self.stable_since >= self.stable_hold_seconds else "MOTION"
                else:
                    self.stable_since = None
                    mode = "MOTION"
        force_full = not ready or bool(self.tracker._weak_candidates)
        if self.risk_scheduler.p["enabled"]:
            mode, force_full, self.risk_status = self.risk_scheduler.choose(
                tracks, previous_positions, stamp, frame.shape, self.minimum_confirmed_hits,
                self.color_roi_bbox_source, self.stable_motion_quality,
                self.motion_speed_threshold, self.stable_hold_seconds,
                weak_candidates=bool(self.tracker._weak_candidates),
                require_light_reference=bool(self.online_parameters["enabled"]))
        live_ids = {t.track_id for t in tracks}
        self.roi_history = {key: entry for key, entry in self.roi_history.items()
                            if key in live_ids and stamp - entry[2] <= self.motion_max_gap}
        for track in tracks:
            if track.visible:
                self.roi_history[track.track_id] = (track.roi_epoch, gray, stamp,
                                                    analysis_bbox(track, self.color_roi_bbox_source))
        # Crop candidates come from this captured frame, never a future track box.
        zoom_boxes = [tuple(analysis_bbox(t, self.color_roi_bbox_source)) for t in tracks
                      if t.visible and t.hits >= self.minimum_confirmed_hits]
        zoom = dict(self.zoom_parameters)
        if self.risk_scheduler.p["enabled"] and mode != "TRACK":
            zoom["full_frame_seconds"] = min(1.0, zoom["full_frame_seconds"])
        context = {"epoch": self.epoch, "boxes": zoom_boxes,
                   "force_full": force_full, "zoom": zoom,
                   "risk": dict(self.risk_status)}
        self.detector_worker.submit(frame, gray, stamp, self.epoch, mode, context=context)
        self.previous_gray, self.previous_sample_time = gray, stamp
        return tracks

    def destroy_node(self):
        worker = getattr(self, "detector_worker", None)
        if worker is not None:
            worker.close()
        return super().destroy_node()

    @staticmethod
    def _header_time(header) -> float:
        timestamp = (
            float(header.stamp.sec)
            + float(header.stamp.nanosec) * 1e-9
        )
        if timestamp > 0.0:
            return timestamp
        return time.monotonic()

    @staticmethod
    def _extract_detections(prediction, shape) -> List[Detection]:
        return detections_from_prediction(prediction, shape)

    def _crop_with_padding(self, frame, bbox):
        x1, y1, x2, y2 = self._padded_bbox(frame.shape, bbox)
        return frame[y1:y2, x1:x2]

    def _padded_bbox(self, shape, bbox):
        height, width = shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = int((x2 - x1) * self.roi_padding_ratio)
        pad_y = int((y2 - y1) * self.roi_padding_ratio)
        return (max(0, x1-pad_x), max(0, y1-pad_y), min(width, x2+pad_x), min(height, y2+pad_y))

    def _track_payload(self, track: Track, now: float) -> dict:
        instant_color = track.instant.label if track.visible else "MISSING"
        instant_confidence = (
            track.instant.confidence if track.visible else 0.0
        )
        zeros = {name: 0.0 for name in ("RED", "GREEN", "BLUE")}
        color_ratios = track.instant.ratios if track.visible else zeros
        raw_ratios = track.instant.raw_ratios if track.visible else zeros
        background_ratios = (
            track.instant.background_ratios if track.visible else zeros
        )
        brightness_contrast = (
            track.instant.brightness_contrast if track.visible else zeros
        )
        channel_shares = (
            track.instant.channel_shares if track.visible else zeros
        )
        output_state = (
            track.decision.committed_state
            if self.state_change_enabled else track.temporal.state
        )
        output_confidence = (
            track.decision.committed_confidence
            if self.state_change_enabled else track.temporal.confidence
        )
        payload = {
            "track_id": track.track_id,
            "visible": track.visible,
            "age_since_seen_s": round(now - track.last_seen, 3),
            "bbox_xyxy": list(track.bbox),
            "detector_confidence": round(track.detector_confidence, 4),
            "association_stage": track.association_stage,
            "steady_off_active": track.steady_off.active,
            "steady_off_reason": track.steady_off.reason,
            "steady_off_observation_s": round(track.steady_off.span, 3),
            "steady_off_signal_range": round(track.steady_off.signal_range, 5),
            "hue_shares": {key.lower(): round(value, 4) for key, value
                           in track.raw_instant.hue_shares.items()} if track.visible else zeros,
            "required_luminance_signal": track.raw_instant.required_signal if track.visible else None,
            "instant_color": instant_color,
            "instant_color_confidence": round(instant_confidence, 3),
            "raw_instant_color": (
                track.raw_instant.label
                if track.visible else "MISSING"
            ),
            "last_reliable_color": track.last_reliable_color,
            "color_ratios": {
                key.lower(): round(value, 6)
                for key, value in color_ratios.items()
            },
            "raw_color_ratios": {
                key.lower(): round(value, 6)
                for key, value in raw_ratios.items()
            },
            "background_color_ratios": {
                key.lower(): round(value, 6)
                for key, value in background_ratios.items()
            },
            "brightness_contrast": {
                key.lower(): round(value, 2)
                for key, value in brightness_contrast.items()
            },
            "channel_shares": {
                key.lower(): round(value, 4)
                for key, value in channel_shares.items()
            },
            "center_value_p90": round(
                track.instant.center_value_p90, 1
            ) if track.visible else 0.0,
            "center_saturation_p75": round(
                track.instant.center_saturation_p75, 1
            ) if track.visible else 0.0,
            "emission_strength_p75": round(
                track.instant.emission_strength_p75, 1
            ) if track.visible else 0.0,
            "background_emission_p75": round(
                track.instant.background_emission_p75, 1
            ) if track.visible else 0.0,
            "emission_contrast": round(
                track.instant.emission_contrast, 1
            ) if track.visible else 0.0,
            "neutral_clip_ratio": round(
                track.instant.neutral_clip_ratio, 4
            ) if track.visible else 0.0,
            "color_value_threshold": round(
                track.instant.color_value_threshold, 1
            ) if track.visible else 0.0,
            "emission_diagnostics": track.raw_instant.emission_diagnostics if track.visible else {},
            "color_decision_reason": (
                track.instant.decision_reason
                if track.visible else "missing"
            ),
            "luminance_signal": round(
                track.instant.luminance_signal, 5
            ) if track.visible else 0.0,
            "blink_reference_ready": track.blink_reference.ready,
            "blink_reference_off_level": (
                round(track.blink_reference.off_level, 5)
                if track.blink_reference.ready else None
            ),
            "blink_reference_on_level": (
                round(track.blink_reference.on_level, 5)
                if track.blink_reference.ready else None
            ),
            "blink_reference_normalized": (
                round(track.blink_reference.normalized_signal, 4)
                if (
                    track.visible
                    and track.blink_reference.ready
                    and math.isfinite(track.blink_reference.normalized_signal)
                ) else None
            ),
            "blink_reference_updates": track.blink_reference.updates,
            "blink_reference_levels_by_color": {
                name.lower(): {"off": round(levels[0], 5), "on": round(levels[1], 5)}
                for name, levels in track.blink_reference.levels_by_color.items()
            },
            "blink_reference_source_state": (
                track.blink_reference.source_state
                if track.blink_reference.ready else None
            ),
            "emitter_center_normalized": [
                round(track.instant.emitter_center_x, 4),
                round(track.instant.emitter_center_y, 4),
            ] if track.visible else [0.5, 0.5],
            "emitter_core_value": round(
                track.instant.emitter_core_value, 1
            ) if track.visible else 0.0,
            "emitter_background_value": round(
                track.instant.emitter_background_value, 1
            ) if track.visible else 0.0,
            "emitter_chroma_fraction": round(
                track.instant.emitter_chroma_fraction, 5
            ) if track.visible else 0.0,
            # V7.2 compatibility aliases use the conservative committed state.
            # The raw temporal diagnostics are still published separately.
            "state": output_state,
            "state_confidence": round(output_confidence, 3),
            "dominant_color": track.temporal.dominant_color,
            "on_ratio": round(track.temporal.on_ratio, 3),
            "transitions": track.temporal.transitions,
            "blink_hz": round(track.temporal.blink_hz, 3),
            "history_span_s": round(track.temporal.history_span, 3),
            "temporal_method": track.temporal.method,
            "template_agreement": round(
                track.temporal.template_agreement, 3
            ),
        }
        if self.state_change_enabled:
            decision = track.decision
            candidate_age = (
                max(0.0, now - decision.candidate_since)
                if decision.candidate_since is not None else None
            )
            committed_age = (
                max(0.0, now - decision.committed_since)
                if decision.committed_since is not None else None
            )
            pending_age = (
                max(0.0, now - decision.pending_since)
                if decision.pending_since is not None else None
            )
            payload.update({
                "raw_temporal_state": track.temporal.state,
                "raw_temporal_confidence": round(
                    track.temporal.confidence, 3
                ),
                "candidate_state": decision.candidate_state,
                "candidate_state_confidence": round(
                    decision.candidate_confidence, 3
                ),
                "candidate_age_s": (
                    round(candidate_age, 3)
                    if candidate_age is not None else None
                ),
                "committed_state": decision.committed_state,
                "committed_state_confidence": round(
                    decision.committed_confidence, 3
                ),
                "committed_age_s": (
                    round(committed_age, 3)
                    if committed_age is not None else None
                ),
                "previous_committed_state": (
                    decision.previous_committed_state
                ),
                "change_pending": decision.change_pending,
                "pending_age_s": (
                    round(pending_age, 3)
                    if pending_age is not None else None
                ),
                "change_detected_at": decision.change_detected_at,
                "confirmed_at": decision.confirmed_at,
                "decision_latency_ms": round(
                    decision.decision_latency_seconds * 1000.0, 1
                ),
                "state_revision": decision.state_revision,
                "state_changed": decision.state_changed,
                "decision_source": decision.source,
                "change_reason": decision.change_reason,
                "change_from_state": (
                    decision.previous_committed_state
                    if decision.state_changed else None
                ),
                "change_to_state": (
                    decision.committed_state
                    if decision.state_changed else None
                ),
            })
        observation = track.raw_instant
        payload.update({
            "photometric_phase": observation.photometric_phase if track.visible else "UNOBSERVED",
            "photometry_mode": observation.photometry_mode,
            "flux_signed": round(observation.flux_signed, 3) if track.visible else None,
            "flux_snr": round(observation.flux_snr, 3) if track.visible else None,
            "spatial_support": round(observation.spatial_support, 4) if track.visible else None,
            "peak_excess": round(observation.peak_excess, 2) if track.visible else None,
            "absolute_color_mass": round(observation.color_mass, 5) if track.visible else None,
            "color_energy": observation.color_energy if track.visible else {},
            "aperture_radius_px": round(observation.aperture_radius_px, 2),
            "emitter_center_source": observation.emitter_center_source,
            "measurement_quality": round(observation.measurement_quality, 3) if track.visible else 0.0,
            "motion_quality": round(track.motion_quality, 3) if track.visible else 0.0,
            "detector_observed_at": track.detector_stamp if self.async_enabled else track.last_seen,
            "roi_observed_at": track.roi_last_observed if track.roi_tracking_active else track.measurement_time,
            "independent_roi_tracking": track.roi_tracking_active,
            "roi_continuity": self.roi_continuity.diagnostics((track.track_id, track.roi_epoch), now),
            "measurement_bbox_xyxy": list(track.measurement_bbox) if track.measurement_bbox else None,
            "light_reference": observation.emission_diagnostics.get("online_light", {}) if track.visible else {},
            "lamp_region": observation.emission_diagnostics.get("lamp_region", {}) if track.visible else {"valid": False, "reason": "beacon_missing"},
            "lamp_measurement_quality": round(self.lamp_measurements.get((track.track_id, track.roi_epoch), observation).measurement_quality, 3) if track.visible else 0.0,
            "periodic_target": self.periodic_targets.get(track.track_id, {
                "priority": False, "periodic_reference_valid": False, "status": "NOT_PERIODICALLY_CONFIRMED"}),
            "state_valid": bool(track.visible and not track.decision.change_pending
                                and observation.emission_diagnostics.get("lamp_region", {}).get("valid", False)
                                and self.lamp_measurements.get((track.track_id, track.roi_epoch), observation).measurement_quality >= self.online_parameters["min_measurement_quality"]
                                and (not self.online_parameters["enabled"] or observation.photometric_phase in ("ON", "OFF"))
                                and output_state != "UNKNOWN" and track.temporal.state == output_state),
            "signal_diagnostics": (track.analyzer.physical_decoder.diagnostics
                                   if track.analyzer.physical_decoder is not None else {}),
        })
        return payload

    def _publish_debug(
        self, source_header, frame, tracks: List[Track], processing_ms: float
    ) -> None:
        if (
            not self.debug_always_publish
            and self.debug_publisher.get_subscription_count() == 0
        ):
            return
        wall_now = time.monotonic()
        if self.debug_max_fps > 0.0:
            interval = 1.0 / self.debug_max_fps
            if self.last_debug_time > 0.0:
                if wall_now - self.last_debug_time < interval:
                    return
                self.last_debug_time += interval
                if self.last_debug_time < wall_now - interval:
                    self.last_debug_time = wall_now
            else:
                self.last_debug_time = wall_now
        else:
            self.last_debug_time = wall_now
        annotated = frame.copy()
        visible_count = 0
        for track in tracks:
            if not track.visible and not self.debug_show_missing_tracks:
                continue
            x1, y1, x2, y2 = track.bbox
            if track.visible:
                visible_count += 1
            phase = track.raw_instant.photometric_phase if track.visible else "UNOBSERVED"
            display_color = (track.raw_instant.label if phase == "ON" else
                             track.temporal.dominant_color if phase == "LEGACY" else "UNKNOWN")
            color = (160, 160, 160) if phase == "OFF" else self._display_color(display_color)
            focus = self.periodic_targets.get(track.track_id, {}).get("priority", False)
            thickness = 4 if focus else (2 if track.visible else 1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            region = self.lamp_regions.get((track.track_id, track.roi_epoch))
            if track.visible and region is not None:
                for ellipse, line_color in ((region.outer, (255, 180, 0)),
                                            (region.aperture, (0, 255, 255))):
                    if ellipse is not None:
                        (cx, cy), (ew, eh), angle = ellipse
                        cv2.ellipse(annotated, (round(cx), round(cy)),
                                    (max(1, round(ew*.5)), max(1, round(eh*.5))),
                                    angle, 0, 360, line_color, 2, cv2.LINE_AA)
            display_state = (
                track.decision.committed_state
                if self.state_change_enabled else track.temporal.state
            )
            candidate_suffix = (
                f" C={track.decision.candidate_state}"
                if self.state_change_enabled else ""
            )
            raw_label = track.raw_instant.label
            effective_label = track.instant.label
            instant_trace = (
                effective_label
                if raw_label == effective_label
                else f"{raw_label}>{effective_label}"
            )
            reference_suffix = (
                f" Bref={track.blink_reference.normalized_signal:.2f}"
                if track.blink_reference.ready else ""
            )
            label = (
                f"ID {track.track_id} {instant_trace}>"
                f"{display_state}{candidate_suffix} "
                f"{track.detector_confidence:.2f} "
                f"{track.association_stage}{reference_suffix} "
                f"S75={track.instant.center_saturation_p75:.0f} "
                f"L={track.instant.luminance_signal:.3f} "
                f"clip={track.instant.neutral_clip_ratio:.2f}"
            )
            phase = track.raw_instant.photometric_phase if track.visible else "UNOBSERVED"
            if not track.visible:
                instant_trace, display_state = "MISSING", "UNKNOWN"
            label = (f"ID {track.track_id}" + (" FOCUS " if focus else " ") + f"{phase}/{instant_trace} {display_state}"
                     + (" HOLD" if track.decision.change_pending else "")
                     + (f" L={track.raw_instant.luminance_signal:.3f}" if track.visible else "")
                     + f" ROI={track.association_stage} q={track.motion_quality:.2f}")
            lamp = track.raw_instant.emission_diagnostics.get("lamp_region", {})
            detail = f"lamp={lamp.get('reason', 'unobserved')} q={lamp.get('quality', 0):.2f} "
            detail += track.raw_instant.decision_reason
            cv2.putText(annotated, detail,
                        (x1, min(annotated.shape[0] - 8, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(
                annotated, label, (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA,
            )
            position = self.frame_positions.get(track.track_id)
            if position is not None:
                if position["valid"]:
                    px, py, pz = position["xyz_m"]
                    text3d = f"XYZ {position['frame_id']}: {px:.2f} {py:.2f} {pz:.2f} m R={position['range_m']:.2f} m"
                    uv = tuple(round(v) for v in position["pixel_uv"])
                    cv2.drawMarker(annotated, uv, (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
                elif position.get("estimate") is not None:
                    estimate = position["estimate"]
                    px, py, pz = estimate["xyz_m"]
                    text3d = f"EST {estimate['frame_id']}: {px:.2f} {py:.2f} {pz:.2f} m (PROVISIONAL)"
                    uv = tuple(round(v) for v in position["pixel_uv"])
                    cv2.drawMarker(annotated, uv, (0, 190, 255), cv2.MARKER_CROSS, 12, 2)
                else:
                    text3d = "3D: " + position["status"]
                cv2.putText(annotated, text3d, (max(0, x1), min(annotated.shape[0]-8, y2+38)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"{self.algorithm_version} beacons={visible_count} "
            f"processing={processing_ms:.1f}ms",
            (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 255, 255), 2, cv2.LINE_AA,
        )
        if self.detector_worker:
            runtime = self.detector_worker.snapshot()
            budget = f"{runtime['duty_budget']:.0%}" if runtime.get("budget_enabled", True) else "OFF"
            cv2.putText(annotated,
                        f"{runtime['schedule']} YOLO={runtime['inference_ms']:.0f}ms "
                        f"idle={runtime['idle_seconds']:.2f}s budget={budget}",
                        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
        if self.debug_image_size is not None:
            if self.localizer.enabled:
                cv2.putText(annotated, "3D " + self.localization_summary["status"][:110],
                            (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            region = self.last_inference_region
            shape_text = (f"input={region['input_shape_hw']}" if region.get("input_shape_hw") else
                          f"target={region.get('letterbox_target_hw', '')}")
            cv2.putText(annotated, f"YOLO {region.get('kind', 'WAITING')} {shape_text} "
                        f"{region.get('reason', '')}", (12, 104), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (0, 255, 255), 1, cv2.LINE_AA)
            annotated = fit_canvas(annotated, self.debug_image_size)
        if self.zoom_parameters["preview_enabled"]:
            items = [(analysis_bbox(t, self.color_roi_bbox_source),
                      f"ID {t.track_id} {t.raw_instant.photometric_phase}/{t.raw_instant.label}",
                      f"L={t.raw_instant.luminance_signal:.3f} {t.association_stage}")
                     for t in tracks if t.visible][:2]
            annotated = add_roi_panel(annotated, frame, items)
        # Draw after preview resizing: these dimensions describe the real input,
        # never the debug canvas or the YOLO letterbox tensor.
        input_text = status_text(self.input_stream.snapshot())
        cv2.putText(annotated, input_text,
                    (12, max(18, annotated.shape[0]-12)), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, input_text,
                    (12, max(18, annotated.shape[0]-12)), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0, 255, 255), 1, cv2.LINE_AA)
        debug_message = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
        debug_message.header = source_header
        self._safe_publish(self.debug_publisher, debug_message)

    @staticmethod
    def _safe_publish(publisher, message) -> bool:
        try:
            publisher.publish(message)
            return True
        except RCLError:
            if rclpy.ok():
                raise
            return False

    @staticmethod
    def _display_color(color: str):
        return {
            "RED": (0, 0, 255),
            "GREEN": (0, 255, 0),
            "BLUE": (255, 100, 0),
        }.get(color, (0, 255, 255))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = Task1BeaconNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
