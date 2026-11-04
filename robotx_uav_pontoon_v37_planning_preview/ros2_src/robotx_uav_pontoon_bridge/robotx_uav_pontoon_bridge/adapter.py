"""Fail-closed normalization layer for the *local* V3.7 detector.

The exact user-local V3.7 Python API has not been supplied. The adapter accepts
common dict/object field layouts, and NEVER fabricates missing keypoints.
Run scripts/probe_v37.py if `v37_schema_unmapped` appears.
"""
import importlib
import inspect
import math
import os
import sys


def get(obj, *names, default=None):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if not isinstance(obj, dict) and hasattr(obj, name):
            return getattr(obj, name)
    return default


def as_uv(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if 'u' in value and 'v' in value:
            value = [value['u'], value['v']]
        elif 'x' in value and 'y' in value:
            value = [value['x'], value['y']]
        elif 'xy' in value:
            value = value['xy']
    elif hasattr(value, 'x') and hasattr(value, 'y'):
        value = [value.x, value.y]
    try:
        u, v = float(value[0]), float(value[1])
        if math.isfinite(u) and math.isfinite(v):
            return [u, v]
    except (TypeError, ValueError, KeyError, IndexError):
        pass
    return None


def point(obj, *names):
    return as_uv(get(obj, *names))


def normalize(raw, width, height, expected_branches=4):
    """Output normalized dict. Validity requires all 3 trunk + 3xN branch points."""
    if isinstance(raw, (tuple, list)) and raw and not isinstance(raw[0], (int, float)):
        raw = raw[0]
    if hasattr(raw, 'to_dict') and callable(raw.to_dict):
        raw = raw.to_dict()
    if not isinstance(raw, dict) and not hasattr(raw, '__dict__'):
        raise ValueError('unexpected_v37_result_type')
    status = str(get(raw, 'status', default='UNKNOWN'))
    detected = bool(get(raw, 'detected', default=(status == 'FULL_PLATFORM')))
    kp = get(raw, 'keypoints', 'keypoint_data', 'keypoint_result', default={}) or {}
    trunk = get(raw, 'trunk', 'trunk_keypoints', default=None) or get(kp, 'trunk', default={}) or {}
    start = (point(raw, 'trunk_start_uv_px', 'trunk_top_uv_px', 'trunk_start_px') or
             point(trunk, 'start', 'top', 'p1', 'start_uv_px'))
    end = (point(raw, 'trunk_end_uv_px', 'trunk_bottom_uv_px', 'trunk_end_px') or
           point(trunk, 'end', 'bottom', 'p2', 'end_uv_px'))
    center = (point(raw, 'trunk_center_uv_px', 'trunk_center_px') or
              point(trunk, 'center', 'center_uv_px'))
    if center is None and start is not None and end is not None:
        center = [(start[0]+end[0])/2, (start[1]+end[1])/2]
    branches_raw = (get(raw, 'branches', 'branch_keypoints', default=None) or
                    get(kp, 'branches', 'branch_keypoints', default=None) or [])
    if isinstance(branches_raw, dict):
        branches_raw = list(branches_raw.values())
    branches = []
    for i, item in enumerate(branches_raw):
        tip = point(item, 'tip_uv_px', 'tip', 'tip_px', 'tip_xy')
        root = point(item, 'root_uv_px', 'root', 'root_px', 'root_xy')
        midpoint = point(item, 'center_uv_px', 'center', 'center_px', 'center_xy')
        if tip is None or root is None or midpoint is None:
            continue
        branches.append({'index': i+1, 'label': f'B{i+1}', 'tip': tip, 'root': root, 'center': midpoint})
    branch_count = int(get(raw, 'branch_count', default=len(branches)) or 0)
    bbox = get(raw, 'bbox_xywh_px', default=[0,0,0,0])
    try:
        bbox = [float(v) for v in bbox]
        if len(bbox) != 4: raise ValueError()
    except (ValueError, TypeError):
        bbox = [0.0]*4
    all_uv = [start, end, center]
    for branch in branches:
        all_uv.extend((branch['tip'], branch['root'], branch['center']))
    in_bounds = all(p is not None and 0 <= p[0] < width and 0 <= p[1] < height for p in all_uv)
    actual_keypoints_valid = (len(branches) == expected_branches and in_bounds)
    source_kp_valid = bool(get(raw, 'keypoints_valid', default=True))
    source_layout = bool(get(raw, 'layout_verified', default=False))
    layout_verified = source_layout and branch_count == expected_branches
    geometry_ready = (status == 'FULL_PLATFORM' and detected and layout_verified and
                      source_kp_valid and actual_keypoints_valid)
    reason = ''
    if not geometry_ready:
        if status != 'FULL_PLATFORM' or not detected: reason = 'not_full_platform'
        elif not source_layout: reason = 'layout_not_verified'
        elif not actual_keypoints_valid: reason = 'v37_schema_unmapped_or_keypoints_invalid'
        else: reason = 'keypoints_not_verified'
    return {
        'status': status, 'detected': detected,
        'geometry_score': float(get(raw, 'geometry_score', default=0.0) or 0.0),
        'branch_count': branch_count, 'layout_verified': layout_verified,
        'keypoints_valid': source_kp_valid and actual_keypoints_valid,
        'geometry_ready': geometry_ready, 'invalid_reason': reason,
        'bbox': bbox, 'trunk_start': start, 'trunk_end': end,
        'trunk_center': center, 'branches': branches,
    }


class V37Adapter:
    def __init__(self, source_dir, max_side=480, expected_branches=4):
        if not source_dir:
            from ament_index_python.packages import get_package_share_directory
            source_dir = os.path.join(get_package_share_directory('robotx_uav_pontoon_bridge'), 'detector')
        source_dir = os.path.abspath(os.path.expanduser(source_dir))
        if not os.path.isfile(os.path.join(source_dir, 'uav_pontoon_v37.py')):
            raise FileNotFoundError(f'uav_pontoon_v37.py missing in {source_dir}')
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)
        module = importlib.import_module('uav_pontoon_v37')
        cls = getattr(module, 'PontoonV37')
        sig = inspect.signature(cls)
        kwargs = {}
        if 'max_side' in sig.parameters: kwargs['max_side'] = max_side
        if 'expected_branches' in sig.parameters: kwargs['expected_branches'] = expected_branches
        self.detector = cls(**kwargs)
        self.expected_branches = expected_branches
        if hasattr(self.detector, 'detect'):
            self.detect_fn = self.detector.detect
        elif hasattr(self.detector, 'process'):
            self.detect_fn = self.detector.process
        elif callable(self.detector):
            self.detect_fn = self.detector
        else:
            raise TypeError('PontoonV37 has no detect/process/call method')

    def detect(self, image_bgr):
        raw = self.detect_fn(image_bgr)
        return normalize(raw, image_bgr.shape[1], image_bgr.shape[0], self.expected_branches)
