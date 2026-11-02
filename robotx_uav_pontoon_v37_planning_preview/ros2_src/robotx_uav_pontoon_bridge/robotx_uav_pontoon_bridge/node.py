"""ROS2 bridge: RGB -> V3.7 UV -> (aligned depth OR measured plane) -> map_ned.

Safety: all coordinate outputs are invalid unless camera model, timestamped TF
and a metric source are valid. This node publishes OBSERVATIONS ONLY.
"""
import json
import math
import traceback
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import Point, Point32
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from robotx_uav_pontoon_interfaces.msg import PontoonObservation, PontoonBranch

from .adapter import V37Adapter
from .geometry import (quat_to_rotation, optical_point_from_depth, transform_point,
                       intersect_plane_z, median_depth_patch)


def p32(uv):
    out = Point32()
    if uv is not None:
        out.x, out.y = float(uv[0]), float(uv[1])
    return out


def pworld(xyz):
    out = Point()
    if xyz is not None:
        out.x, out.y, out.z = [float(x) for x in xyz]
    return out


def time_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class PontoonNode(Node):
    def __init__(self):
        super().__init__('uav_pontoon_v37_bridge')
        defaults = {
            'source_dir': '/home/yangxy/robotx_ws/src/robotx_perception/robotx_uav_perception/robotx_uav_perception/task3_pontoon',
            'image_topic': '/uav/camera/image_rect_color',
            'camera_info_topic': '/uav/camera/camera_info',
            'aligned_depth_topic': '/uav/camera/aligned_depth_to_color/image_raw',
            'observation_topic': '/robotx/perception/uav/pontoon/observation',
            'status_topic': '/robotx/perception/uav/pontoon/status',
            'world_frame': 'map_ned',
            'max_side': 480,
            'expected_branches': 4,
            'use_aligned_depth': True,
            'max_depth_age_sec': 0.08,
            'enable_plane_projection': False,
            'plane_height_verified': False,
            'plane_z_ned_m': 0.0,
            'max_plane_range_m': 50.0,
            'branch_ids_verified': False,
            'uv_coordinates_verified': False,
            'metric_calibration_verified': False,
            'log_every_n_frames': 30,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.cfg = {key: self.get_parameter(key).value for key in defaults}
        self.bridge = CvBridge()
        self.camera_info = None
        self.depth = None
        self.depth_stamp = None
        self.depth_frame = None
        self.count = 0
        self.detector = V37Adapter(self.cfg['source_dir'], self.cfg['max_side'], self.cfg['expected_branches'])
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        image_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                               reliability=ReliabilityPolicy.BEST_EFFORT)
        self.obs_pub = self.create_publisher(PontoonObservation, self.cfg['observation_topic'], 10)
        self.status_pub = self.create_publisher(String, self.cfg['status_topic'], 10)
        self.image_sub = self.create_subscription(Image, self.cfg['image_topic'], self.on_image, image_qos)
        self.info_sub = self.create_subscription(CameraInfo, self.cfg['camera_info_topic'], self.on_info, image_qos)
        if self.cfg['use_aligned_depth']:
            self.depth_sub = self.create_subscription(Image, self.cfg['aligned_depth_topic'], self.on_depth, image_qos)
        self.get_logger().info('V3.7 bridge started; all world points invalid until calibrated TF + metric source exist')
        if self.cfg['enable_plane_projection'] and not self.cfg['plane_height_verified']:
            self.get_logger().warn('Plane projection requested but plane_height_verified=false; projection DISABLED')
        if self.cfg['branch_ids_verified']:
            self.get_logger().warn('branch_ids_verified=true: operator is responsible for checking physical B1..B4 mapping')

    def on_info(self, msg):
        self.camera_info = msg

    def on_depth(self, msg):
        try:
            if msg.encoding == '16UC1':
                depth_m = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32) * 0.001
            elif msg.encoding == '32FC1':
                depth_m = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32)
            else:
                self.depth = None
                return
            if depth_m.ndim != 2:
                self.depth = None
                return
            self.depth, self.depth_stamp, self.depth_frame = depth_m, msg.header.stamp, msg.header.frame_id
        except Exception as exc:
            self.depth = None
            self.get_logger().warn(f'depth decode failed: {exc}')

    def camera_model(self, image_msg):
        info = self.camera_info
        if info is None:
            raise ValueError('camera_info_missing')
        if info.width != image_msg.width or info.height != image_msg.height:
            raise ValueError('camera_info_image_size_mismatch')
        if not image_msg.header.frame_id or info.header.frame_id != image_msg.header.frame_id:
            raise ValueError('camera_info_optical_frame_mismatch')
        # Input must be rectified. CameraInfo.P models the rectified image.
        fx, fy, cx, cy = float(info.p[0]), float(info.p[5]), float(info.p[2]), float(info.p[6])
        if not all(math.isfinite(x) for x in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
            raise ValueError('rectified_camera_info_P_invalid')
        return fx, fy, cx, cy

    def metric_context(self, image_msg):
        """Build metric context at image timestamp. Never use latest TF fallback."""
        fx, fy, cx, cy = self.camera_model(image_msg)
        if time_seconds(image_msg.header.stamp) <= 0:
            raise ValueError('image_timestamp_invalid')
        tf = self.tf_buffer.lookup_transform(
            self.cfg['world_frame'], image_msg.header.frame_id,
            Time.from_msg(image_msg.header.stamp), timeout=Duration(seconds=0.08))
        q = tf.transform.rotation
        R = quat_to_rotation(q.x, q.y, q.z, q.w)
        tr = tf.transform.translation
        T = np.array([tr.x, tr.y, tr.z], dtype=float)
        if not np.all(np.isfinite(T)):
            raise ValueError('invalid_tf_translation')
        valid_depth = (self.cfg['use_aligned_depth'] and self.depth is not None and
                       self.depth.shape == (image_msg.height, image_msg.width) and
                       self.depth_frame == image_msg.header.frame_id and
                       self.depth_stamp is not None and
                       abs(time_seconds(self.depth_stamp) - time_seconds(image_msg.header.stamp))
                       <= float(self.cfg['max_depth_age_sec']))
        plane_ok = self.cfg['enable_plane_projection'] and self.cfg['plane_height_verified']
        return (fx, fy, cx, cy, R, T, valid_depth, plane_ok)

    def uv_to_world(self, uv, ctx):
        if uv is None:
            return None, 'none'
        fx, fy, cx, cy, R, T, depth_ok, plane_ok = ctx
        if depth_ok:
            d = median_depth_patch(self.depth, uv[0], uv[1])
            if d is not None:
                xyz = transform_point(optical_point_from_depth(
                    uv[0], uv[1], d, fx, fy, cx, cy), R, T)
                return xyz, 'aligned_depth'
        if plane_ok:
            try:
                xyz = intersect_plane_z(uv[0], uv[1], fx, fy, cx, cy, R, T,
                    float(self.cfg['plane_z_ned_m']), max_range_m=float(self.cfg['max_plane_range_m']))
                return xyz, 'known_plane'
            except ValueError:
                pass
        return None, 'none'

    def on_image(self, image_msg):
        self.count += 1
        obs = PontoonObservation()
        obs.header = image_msg.header
        obs.world_frame_id = self.cfg['world_frame']
        obs.detector_version = 'V3.7'
        obs.status = 'UNKNOWN'
        obs.invalid_reason = 'detector_not_run'
        obs.image_width, obs.image_height = image_msg.width, image_msg.height
        obs.xyz_method = 'none'
        obs.branch_ids_verified = bool(self.cfg['branch_ids_verified'])
        obs.uv_coordinates_verified = bool(self.cfg['uv_coordinates_verified'])
        obs.metric_calibration_verified = bool(self.cfg['metric_calibration_verified'])
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            result = self.detector.detect(frame)
            obs.status = result['status']
            obs.detected = result['detected']
            obs.geometry_score = result['geometry_score']
            obs.branch_count = result['branch_count']
            obs.layout_verified = result['layout_verified']
            obs.keypoints_valid = result['keypoints_valid']
            obs.geometry_ready = result['geometry_ready']
            obs.invalid_reason = result['invalid_reason']
            obs.bbox_xywh_px = result['bbox']
            obs.trunk_start_uv_px = p32(result['trunk_start'])
            obs.trunk_end_uv_px = p32(result['trunk_end'])
            obs.trunk_center_uv_px = p32(result['trunk_center'])
            for b in result['branches']:
                branch = PontoonBranch()
                branch.index, branch.label = b['index'], b['label']
                branch.tip_uv_px = p32(b['tip'])
                branch.root_uv_px = p32(b['root'])
                branch.center_uv_px = p32(b['center'])
                obs.branches.append(branch)
            if obs.geometry_ready:
                try:
                    ctx = self.metric_context(image_msg)
                    methods = set()
                    required = []
                    for key in ('trunk_start', 'trunk_end', 'trunk_center'):
                        xyz, method = self.uv_to_world(result[key], ctx)
                        setattr(obs, f'{key}_world_m', pworld(xyz))
                        setattr(obs, f'{key}_world_valid', xyz is not None)
                        required.append(xyz is not None)
                        if xyz is not None: methods.add(method)
                    for branch, b in zip(obs.branches, result['branches']):
                        for key in ('tip', 'root', 'center'):
                            xyz, method = self.uv_to_world(b[key], ctx)
                            setattr(branch, f'{key}_world_m', pworld(xyz))
                            setattr(branch, f'{key}_world_valid', xyz is not None)
                            required.append(xyz is not None)
                            if xyz is not None: methods.add(method)
                    obs.xyz_valid = bool(required) and all(required)
                    obs.xyz_method = (next(iter(methods)) if len(methods) == 1 else
                                      'mixed' if len(methods) > 1 else 'none')
                    if not obs.xyz_valid:
                        obs.invalid_reason = 'metric_points_incomplete_or_depth_invalid'
                except Exception as exc:
                    obs.xyz_valid = False
                    obs.xyz_method = 'none'
                    obs.invalid_reason = f'metric_unavailable:{type(exc).__name__}:{str(exc)[:120]}'
            obs.planning_ready = bool(obs.geometry_ready and obs.xyz_valid and obs.branch_ids_verified
                                      and obs.uv_coordinates_verified and obs.metric_calibration_verified)
            if obs.planning_ready:
                obs.invalid_reason = ''
            elif obs.geometry_ready and obs.xyz_valid:
                missing = []
                if not obs.branch_ids_verified: missing.append('physical_branch_ids')
                if not obs.uv_coordinates_verified: missing.append('uv_pixel_coordinates')
                if not obs.metric_calibration_verified: missing.append('metric_calibration')
                obs.invalid_reason = 'unverified:' + ','.join(missing)
        except Exception as exc:
            obs.status = 'UNKNOWN'
            obs.detected = False
            obs.geometry_ready = False
            obs.xyz_valid = False
            obs.planning_ready = False
            obs.invalid_reason = f'detector_error:{type(exc).__name__}:{str(exc)[:120]}'
            self.get_logger().error(traceback.format_exc()[-1000:])
        self.obs_pub.publish(obs)
        s = String()
        s.data = json.dumps({
            'status': obs.status, 'detected': obs.detected,
            'branches': obs.branch_count, 'geometry_ready': obs.geometry_ready,
            'xyz_valid': obs.xyz_valid, 'planning_ready': obs.planning_ready,
            'reason': obs.invalid_reason, 'image_stamp': time_seconds(obs.header.stamp),
        }, ensure_ascii=True, allow_nan=False)
        self.status_pub.publish(s)
        if self.count % max(1, int(self.cfg['log_every_n_frames'])) == 0:
            self.get_logger().info(s.data)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PontoonNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
