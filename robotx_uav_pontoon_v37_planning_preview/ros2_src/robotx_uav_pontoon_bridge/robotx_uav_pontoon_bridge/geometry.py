"""Pure math for pixel -> optical -> world, without ROS imports.

Optical convention: +X right, +Y down, +Z forward.
World frame supplied by TF (RobotX: map_ned, +X north +Y east +Z down).
No assumed fixed camera mounting or map origin.
"""
import math
import numpy as np


def pixel_ray(u, v, fx, fy, cx, cy):
    values = (u, v, fx, fy, cx, cy)
    if not all(math.isfinite(float(x)) for x in values) or fx <= 0 or fy <= 0:
        raise ValueError('invalid_camera_intrinsics_or_uv')
    return np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)


def quat_to_rotation(x, y, z, w):
    q = np.array([x, y, z, w], dtype=float)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        raise ValueError('invalid_tf_quaternion')
    x, y, z, w = q / n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=float)


def optical_point_from_depth(u, v, depth_m, fx, fy, cx, cy):
    if not math.isfinite(float(depth_m)) or depth_m <= 0:
        raise ValueError('invalid_depth')
    return pixel_ray(u, v, fx, fy, cx, cy) * depth_m


def transform_point(point_optical, rotation_world_from_optical, translation_world):
    p = rotation_world_from_optical @ np.asarray(point_optical, dtype=float) + np.asarray(translation_world, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError('nonfinite_world_point')
    return p


def intersect_plane_z(u, v, fx, fy, cx, cy, rotation_world_from_optical,
                      translation_world, plane_z_world, min_cos=0.15,
                      max_range_m=50.0):
    """Ray intersection with horizontal z=plane_z_world; returns world XYZ.

    plane_z_world must be a measured/verified pontoon TOP height in map_ned.
    Rejects intersections behind camera, near-parallel rays and excessive range.
    """
    origin = np.asarray(translation_world, dtype=float)
    direction = rotation_world_from_optical @ pixel_ray(u, v, fx, fy, cx, cy)
    norm = float(np.linalg.norm(direction))
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction)):
        raise ValueError('invalid_ray_or_tf')
    if not math.isfinite(float(plane_z_world)):
        raise ValueError('plane_height_unset')
    if norm < 1e-8 or abs(direction[2]) / norm < min_cos:
        raise ValueError('ray_parallel_to_plane')
    t = (float(plane_z_world) - origin[2]) / direction[2]
    if t <= 0 or t * norm > max_range_m:
        raise ValueError('plane_intersection_out_of_range')
    return transform_point(t * pixel_ray(u, v, fx, fy, cx, cy), rotation_world_from_optical, origin)


def median_depth_patch(depth_m, u, v, half_width=2, min_samples=5,
                       min_m=0.25, max_m=30.0, max_spread_m=0.30):
    """Aligned depth array must be in meters and registered to input RGB pixels."""
    arr = np.asarray(depth_m)
    if arr.ndim != 2 or not np.isfinite([u, v]).all():
        return None
    x, y = int(round(u)), int(round(v))
    if x < 0 or y < 0 or x >= arr.shape[1] or y >= arr.shape[0]:
        return None
    crop = arr[max(0,y-half_width):min(arr.shape[0],y+half_width+1),
               max(0,x-half_width):min(arr.shape[1],x+half_width+1)]
    good = crop[np.isfinite(crop) & (crop >= min_m) & (crop <= max_m)]
    if good.size < min_samples:
        return None
    med = float(np.median(good))
    # Reject strong discontinuities at float edge; don't interpolate false geometry.
    if float(np.percentile(good, 90) - np.percentile(good, 10)) > max_spread_m:
        return None
    return med
