"""Publish native and legacy preview images, with measured stream status."""

import json
import time

import cv2

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from .camera_geometry import CameraCalibration
from .v4l2_current_capture import CurrentModeCapture


def load_measured_calibration(filename, width, height):
    if not filename:
        return None
    calibration = CameraCalibration(filename)
    if calibration.approximate:
        raise ValueError("Nominal calibration is not a measured CameraInfo")
    calibration.matrix(width, height, allow_resize=False)
    return calibration


class ArducamCameraNode(Node):
    def __init__(self):
        super().__init__("arducam_camera")
        self.capture = None
        try:
            for name, default in (("video_device", "/dev/video0"),
                                  ("camera_mode", "max"),
                                  ("camera_frame_id", "arducam_optical_frame"),
                                  ("image_topic", "/arducam/image_raw"),
                                  ("native_image_topic", "/arducam/image_native"),
                                  ("camera_info_topic", "/arducam/camera_info"),
                                  ("native_camera_info_topic", "/arducam/camera_info_native"),
                                  ("status_topic", "/arducam/status_json"),
                                  ("publish_image_size", [960, 600]),
                                  ("publish_fps", 5.0),
                                  ("calibration_file", "")):
                self.declare_parameter(name, default)
            self.frame_id = str(self.get_parameter("camera_frame_id").value)
            self.publish_size = list(self.get_parameter("publish_image_size").value)
            self.publish_fps = float(self.get_parameter("publish_fps").value)
            if (len(self.publish_size) != 2 or min(self.publish_size) <= 0
                    or self.publish_fps <= 0):
                raise ValueError("Preview size and publish_fps must be positive")
            self.last_preview = float("-inf")
            self.capture = CurrentModeCapture(
                str(self.get_parameter("video_device").value),
                camera_mode=str(self.get_parameter("camera_mode").value))
            self.status_start = time.monotonic()
            self.status_capture_start = self.capture.dequeued_frame_count
            self.native_count = self.preview_count = 0
            self.status_native_start = self.status_preview_start = 0
            self.last_preview_size = None
            self.calibration = None
            try:
                self.calibration = load_measured_calibration(
                    str(self.get_parameter("calibration_file").value),
                    self.capture.width, self.capture.height)
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.get_logger().warning("CameraInfo not published: " + str(error))
            qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                             reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.images = self.create_publisher(
                Image, str(self.get_parameter("image_topic").value), qos)
            self.native_images = self.create_publisher(
                Image, str(self.get_parameter("native_image_topic").value), qos)
            self.infos = self.create_publisher(
                CameraInfo, str(self.get_parameter("camera_info_topic").value), qos)
            self.native_infos = self.create_publisher(
                CameraInfo, str(self.get_parameter("native_camera_info_topic").value), qos)
            self.status = self.create_publisher(
                String, str(self.get_parameter("status_topic").value), 10)
            rate = "unknown" if self.capture.fps is None else "%.2f" % self.capture.fps
            self.get_logger().info(
                "Arducam %s mode: %dx%d %s; negotiated nominal %s FPS "
                "(not measured); exposure/WB/gain controls are untouched"
                % (self.capture.camera_mode, self.capture.width, self.capture.height,
                   self.capture.pixel_format, rate))
            for warning in self.capture.mode_warnings:
                self.get_logger().warning("Camera mode fallback: " + warning)
        except BaseException:
            if self.capture is not None:
                self.capture.close()
            self.destroy_node()
            raise

    def run(self):
        while rclpy.ok():
            try:
                frame = self.capture.read()
            except TimeoutError:
                if not rclpy.ok():
                    return
                raise
            if not rclpy.ok():
                return
            if frame is None:
                continue
            # ROS receipt time, not a claimed hardware-trigger timestamp.
            stamp = self.get_clock().now().to_msg()
            self.publish_frame(frame, stamp, time.monotonic())

    def publish_frame(self, frame, stamp, now):
        native = self._image(frame, stamp)
        self.native_images.publish(native)
        self.native_count += 1
        if self.calibration is not None:
            self.native_infos.publish(self._info(native))
        if now - self.last_preview < 1.0 / self.publish_fps:
            self._publish_status(now)
            return
        self.last_preview = now
        height, width = frame.shape[:2]
        scale = min(self.publish_size[0] / width, self.publish_size[1] / height, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (max(1, round(width * scale)),
                                      max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
        preview = self._image(frame, stamp)
        self.images.publish(preview)
        self.preview_count += 1
        self.last_preview_size = [preview.width, preview.height]
        if self.calibration is not None:
            self.infos.publish(self._info(preview))
        self._publish_status(now)

    def _publish_status(self, now):
        elapsed = now - self.status_start
        if elapsed < 1.0:
            return
        capture_count = self.capture.dequeued_frame_count
        capture_fps = (capture_count - self.status_capture_start) / elapsed
        native_fps = (self.native_count - self.status_native_start) / elapsed
        preview_fps = (self.preview_count - self.status_preview_start) / elapsed
        requested = self.capture.requested_mode
        data = {
            "schema_version": 1, "camera_mode": self.capture.camera_mode,
            "native_width": self.capture.width, "native_height": self.capture.height,
            "pixel_format": self.capture.pixel_format,
            "negotiated_nominal_fps": self.capture.fps,
            "capture_fps": capture_fps, "native_publish_fps": native_fps,
            "preview_publish_fps": preview_fps, "preview_size": self.last_preview_size,
            "measurement_window_s": elapsed,
            "capture_rate_source": "successfully_dequeued_native_buffers",
            "capture_frames_total": capture_count, "native_frames_total": self.native_count,
            "preview_frames_total": self.preview_count,
            "native_topic": str(self.get_parameter("native_image_topic").value),
            "preview_topic": str(self.get_parameter("image_topic").value),
            "requested_max_mode": (None if requested is None else {
                "width": requested.width, "height": requested.height,
                "pixel_format": requested.pixel_format, "nominal_fps": float(requested.fps)}),
            "mode_fallbacks": self.capture.mode_warnings,
        }
        self.status.publish(String(data=json.dumps(data, ensure_ascii=False, allow_nan=False)))
        self.get_logger().info(
            "Arducam native %dx%d %s | measured capture %.2f FPS, publish %.2f FPS "
            "| preview %.2f FPS | negotiated nominal %s FPS (not YOLO FPS)"
            % (self.capture.width, self.capture.height, self.capture.pixel_format,
               capture_fps, native_fps, preview_fps,
               "unknown" if self.capture.fps is None else "%.2f" % self.capture.fps))
        self.status_start = now
        self.status_capture_start = capture_count
        self.status_native_start, self.status_preview_start = self.native_count, self.preview_count

    def _image(self, frame, stamp):
        message = Image()
        message.header.stamp, message.header.frame_id = stamp, self.frame_id
        message.height, message.width = frame.shape[:2]
        message.encoding = "bgr8"
        message.is_bigendian = 0
        message.step = message.width * 3
        message.data = frame.tobytes()
        return message

    def _info(self, image):
        info = CameraInfo()
        info.header = image.header
        info.width, info.height = image.width, image.height
        self.calibration.fill_info(info, image.width, image.height)
        return info


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ArducamCameraNode()
        node.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.capture.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
