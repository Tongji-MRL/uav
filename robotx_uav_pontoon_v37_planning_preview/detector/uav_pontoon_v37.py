
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RobotX UAV Pontoon V3.7
Conservative Trunk Endpoint Refinement

依赖同目录下：
uav_pontoon_v32.py
uav_pontoon_v34.py
uav_pontoon_v35.py
uav_pontoon_v36.py

功能：
1. 保留 V3.6 的浮台识别与分支关键点
2. 利用分支之间的空隙测量主干左右边界
3. 鲁棒直线拟合主干轴线
4. 重新估计主干两端
5. 失败时回退 V3.6
6. 检查分支长度异常，但不擅自修改分支坐标

只输出图像 UV 坐标。
不输出世界 XYZ。
不接入 UAV 飞控。
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


def robust_fit(y, x):
    y = np.asarray(y, np.float64)
    x = np.asarray(x, np.float64)

    if len(y) < 12 or np.ptp(y) < 20:
        return None

    keep = np.ones(len(y), bool)

    for _ in range(4):
        if keep.sum() < 12:
            return None

        line = np.polyfit(y[keep], x[keep], 1)

        errors = np.abs(
            x - np.polyval(line, y)
        )

        mad = np.median(errors[keep])

        threshold = max(
            1.6,
            3.0 * 1.4826 * mad
        )

        new_keep = errors <= threshold

        if new_keep.sum() < 12:
            return None

        if np.array_equal(keep, new_keep):
            break

        keep = new_keep

    line = np.polyfit(
        y[keep],
        x[keep],
        1
    )

    median_error = float(
        np.median(
            np.abs(
                x[keep]
                - np.polyval(line, y[keep])
            )
        )
    )

    return line, median_error, int(keep.sum())


def refine_trunk(
    mask_small,
    image_shape,
    geometry,
    old_top,
    old_bottom
):
    """
    根据两侧实测边界估计主干两端。

    返回：
    (refined_dict, "ok")

    或：
    (None, failure_reason)
    """

    if (
        not geometry.get("passed")
        or geometry.get("scan_angle_deg") is None
    ):
        return None, "missing_full_geometry"

    ys, xs = np.where(mask_small > 0)

    if len(xs) < 120:
        return None, "empty_mask"

    x0 = int(xs.min())
    y0 = int(ys.min())

    bw = int(xs.max()) - x0 + 1
    bh = int(ys.max()) - y0 + 1

    roi = (
        mask_small[
            y0:y0 + bh,
            x0:x0 + bw
        ] > 0
    ).astype(np.uint8)

    # 与 V3.5 / V3.6 保持一致
    factor = 175.0 / max(roi.shape)

    rw = max(1, round(bw * factor))
    rh = max(1, round(bh * factor))

    roi = cv2.resize(
        roi,
        (rw, rh),
        interpolation=cv2.INTER_NEAREST
    )

    # ==================================================
    # 1. 旋转到标准几何方向
    # ==================================================

    size = int(
        math.ceil(math.hypot(rh, rw))
    ) + 4

    canvas = np.zeros(
        (size, size),
        np.uint8
    )

    pad_x = (size - rw) // 2
    pad_y = (size - rh) // 2

    canvas[
        pad_y:pad_y + rh,
        pad_x:pad_x + rw
    ] = roi

    mat = cv2.getRotationMatrix2D(
        ((size - 1) / 2, (size - 1) / 2),
        float(geometry["scan_angle_deg"]),
        1.0
    )

    rotated = cv2.warpAffine(
        canvas,
        mat,
        (size, size),
        flags=cv2.INTER_NEAREST
    )

    yy, xx = np.where(rotated > 0)

    if len(xx) == 0:
        return None, "empty_aligned"

    crop_x = int(xx.min())
    crop_y = int(yy.min())

    aligned = rotated[
        crop_y:int(yy.max()) + 1,
        crop_x:int(xx.max()) + 1
    ]

    flipped = (
        geometry.get("trunk_aligned_side")
        == "left"
    )

    if flipped:
        aligned = aligned[:, ::-1]

    h, w = aligned.shape

    if h < 70 or w < 25:
        return None, "aligned_too_small"

    # ==================================================
    # 2. 选取分支之间的空隙行
    # ==================================================

    arm_a = max(0, round(.10 * w))
    arm_b = min(w, round(.59 * w))

    if arm_b <= arm_a:
        return None, "invalid_arm_window"

    gap_profile = aligned[
        :,
        arm_a:arm_b
    ].mean(axis=1)

    fit_y = []
    fit_l = []
    fit_r = []

    for y in range(2, h - 2):

        # 分支占有率较高的行不用于拟合
        if gap_profile[y] > .14:
            continue

        candidates = np.flatnonzero(
            aligned[y] > 0
        )

        candidates = candidates[
            candidates >= round(.54 * w)
        ]

        if len(candidates) < max(
            5, int(.10 * w)
        ):
            continue

        left, right = np.percentile(
            candidates,
            [2, 98]
        )

        width = right - left

        if not (
            .12 * w < width < .46 * w
        ):
            continue

        fit_y.append(y)
        fit_l.append(left)
        fit_r.append(right)

    if len(fit_y) < max(
        12, int(.13 * h)
    ):
        return None, "insufficient_trunk_gap_rows"

    # ==================================================
    # 3. 分别拟合主干两条边界
    # ==================================================

    left_fit = robust_fit(
        fit_y, fit_l
    )

    right_fit = robust_fit(
        fit_y, fit_r
    )

    if left_fit is None or right_fit is None:
        return None, "trunk_edge_fit_failed"

    lf, lres, _ = left_fit
    rf, rres, _ = right_fit

    if max(lres, rres) > max(
        2.6, .045 * w
    ):
        return None, "trunk_edges_unstable"

    # ==================================================
    # 4. 测量主干实际覆盖的纵向区间
    # ==================================================

    occupancy = np.zeros(
        h,
        np.float32
    )

    for y in range(h):

        left = float(
            np.polyval(lf, y)
        )

        right = float(
            np.polyval(rf, y)
        )

        width = right - left

        if not (
            .10 * w <= width <= .49 * w
        ):
            continue

        a = max(
            0,
            int(round(left + .12 * width))
        )

        b = min(
            w,
            int(round(right - .12 * width))
        )

        if b - a >= 3:
            occupancy[y] = float(
                aligned[y, a:b].mean()
            )

    flags = (
        occupancy > .42
    ).astype(np.uint8).reshape(h, 1)

    # 补偿模块拼接产生的少量横向接缝
    flags = cv2.morphologyEx(
        flags,
        cv2.MORPH_CLOSE,
        np.ones((7, 1), np.uint8)
    ).reshape(-1)

    delta = np.diff(
        np.r_[0, flags, 0].astype(np.int16)
    )

    runs = [
        (int(a), int(b))
        for a, b in zip(
            np.where(delta == 1)[0],
            np.where(delta == -1)[0]
        )
    ]

    if not runs:
        return None, "trunk_no_supported_rows"

    start, end = max(
        runs,
        key=lambda z: z[1] - z[0]
    )

    if end - start < .68 * h:
        return None, "trunk_span_too_short"

    top_y = float(start)
    bottom_y = float(end - 1)

    # ==================================================
    # 5. 逆变换到原始图像坐标
    # ==================================================

    inv = cv2.invertAffineTransform(mat)

    sh, sw = mask_small.shape
    oh, ow = image_shape[:2]

    def to_original(x, y):

        if flipped:
            x = w - 1 - x

        p = inv @ np.array(
            [
                x + crop_x,
                y + crop_y,
                1.0
            ],
            np.float64
        )

        u = (
            (
                (p[0] - pad_x) / factor
                + x0
            ) * ow / sw
        )

        v = (
            (
                (p[1] - pad_y) / factor
                + y0
            ) * oh / sh
        )

        return np.array(
            [u, v],
            np.float64
        )

    def centerline_x(y):
        return float(
            .5 * (
                np.polyval(lf, y)
                + np.polyval(rf, y)
            )
        )

    new_a = to_original(
        centerline_x(top_y),
        top_y
    )

    new_b = to_original(
        centerline_x(bottom_y),
        bottom_y
    )

    old_a = np.array(
        old_top,
        np.float64
    )

    old_b = np.array(
        old_bottom,
        np.float64
    )

    # 保持旧版本主干两端的方向约定
    direct_cost = (
        np.linalg.norm(new_a - old_a)
        + np.linalg.norm(new_b - old_b)
    )

    reverse_cost = (
        np.linalg.norm(new_b - old_a)
        + np.linalg.norm(new_a - old_b)
    )

    if reverse_cost < direct_cost:
        new_a, new_b = new_b, new_a

    shifts = [
        float(np.linalg.norm(new_a - old_a)),
        float(np.linalg.norm(new_b - old_b))
    ]

    # ==================================================
    # 6. 几何保护，异常时回退
    # ==================================================

    diag = math.hypot(ow, oh)

    if max(shifts) > min(
        40.0, .075 * diag
    ):
        return None, "excessive_endpoint_shift"

    prev_span = float(
        np.linalg.norm(old_b - old_a)
    )

    new_span = float(
        np.linalg.norm(new_b - new_a)
    )

    if not (
        .75 * prev_span
        < new_span
        < 1.25 * prev_span
    ):
        return None, "trunk_length_inconsistent"

    for p in (new_a, new_b):
        if not (
            0 <= p[0] < ow
            and 0 <= p[1] < oh
        ):
            return None, "trunk_point_out_of_image"

    center = .5 * (new_a + new_b)

    angle_deg = math.degrees(
        math.atan2(
            new_b[1] - new_a[1],
            new_b[0] - new_a[0]
        )
    )

    return {
        "trunk_top_uv_px": [
            round(float(z), 1)
            for z in new_a
        ],

        "trunk_bottom_uv_px": [
            round(float(z), 1)
            for z in new_b
        ],

        "trunk_center_uv_px": [
            round(float(z), 1)
            for z in center
        ],

        "trunk_angle_image_deg":
            round(float(angle_deg), 2),

        "trunk_edge_fit_median_residual_px_aligned":
            round(max(lres, rres), 3),

        "trunk_endpoint_shifts_px": [
            round(z, 2)
            for z in shifts
        ],

        "trunk_support_rows":
            int(end - start)
    }, "ok"


# ============================================================
# 7. 分支长度异常提示（不修改分支坐标）
# ============================================================

def analyze_branch_lengths(branches):

    if len(branches) < 3:
        return []

    lengths = np.array([
        np.linalg.norm(
            np.asarray(b["tip_uv_px"])
            - np.asarray(b["root_uv_px"])
        )
        for b in branches
    ], dtype=float)

    median = float(
        np.median(lengths)
    )

    if median <= 1:
        return []

    warnings = []

    for b, length in zip(branches, lengths):

        ratio = length / median

        if not (.70 <= ratio <= 1.35):
            warnings.append({
                "id": b["id"],
                "length_ratio": round(
                    float(ratio), 3
                )
            })

    return warnings


# ============================================================
# 8. V3.7 主检测器
# ============================================================

class PontoonV37:

    def __init__(
        self,
        max_side=480,
        expected_branches=4
    ):

        from uav_pontoon_v36 import PontoonV36

        self.base = PontoonV36(
            max_side=max_side,
            expected_branches=expected_branches
        )

    def detect(self, frame):

        t0 = time.perf_counter()

        info, mask = self.base.detect(frame)

        info = dict(info)

        info["trunk_refinement_applied"] = False
        info["trunk_refinement_reason"] = (
            "no_verified_layout"
        )

        info["tip_length_warnings"] = []
        info["xyz_valid"] = False

        if not (
            info.get("layout_verified")
            and info.get("keypoints_valid")
        ):
            info["processing_ms"] = round(
                (time.perf_counter() - t0) * 1000,
                1
            )

            return info, mask

        info["tip_length_warnings"] = (
            analyze_branch_lengths(
                info.get("branches", [])
            )
        )

        refined, reason = refine_trunk(
            mask,
            frame.shape,
            info.get("geometry_details", {}),
            info["trunk_top_uv_px"],
            info["trunk_bottom_uv_px"]
        )

        info["trunk_refinement_reason"] = reason

        if refined is not None:
            info.update(refined)
            info["trunk_refinement_applied"] = True

        info["processing_ms"] = round(
            (time.perf_counter() - t0) * 1000,
            1
        )

        return info, mask


# ============================================================
# 9. CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    modes = parser.add_mutually_exclusive_group(
        required=True
    )

    modes.add_argument("--image")

    modes.add_argument(
        "--self-test",
        action="store_true"
    )

    parser.add_argument("--output")

    parser.add_argument(
        "--max-side",
        type=int,
        default=480
    )

    args = parser.parse_args()

    if args.max_side < 180:
        parser.error(
            "--max-side must be >= 180"
        )

    detector = PontoonV37(
        max_side=args.max_side
    )

    if args.self_test:

        from uav_pontoon_v32 import synthetic

        good = 0

        cases = [
            ("blue_0", (150, 75, 25), 0),
            ("gray_37", (125, 125, 125), 37),
            ("brown_82", (80, 105, 130), 82)
        ]

        for name, color, angle in cases:

            frame = synthetic(
                color, angle
            )

            result, _ = detector.detect(frame)

            ok = bool(
                result.get("layout_verified")
                and len(result.get("branches", [])) == 4
            )

            good += int(ok)

            print(
                name,
                "PASS" if ok else "FAIL",
                "trunk_refined=",
                result["trunk_refinement_applied"],
                "reason=",
                result["trunk_refinement_reason"]
            )

        print(f"SELF TEST: {good}/3")

        raise SystemExit(
            0 if good == 3 else 1
        )

    frame = cv2.imread(args.image)

    if frame is None:
        parser.error(
            "Cannot open image: " + args.image
        )

    result, mask = detector.detect(frame)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )

    from uav_pontoon_v35 import visualize

    destination = (
        args.output
        or str(
            Path(args.image).with_name(
                Path(args.image).stem
                + "_pontoon_v37.jpg"
            )
        )
    )

    debug = visualize(
        frame,
        result,
        mask
    )

    if not cv2.imwrite(destination, debug):
        raise RuntimeError(
            "Cannot save " + destination
        )

    print("DEBUG_IMAGE:", destination)


if __name__ == "__main__":
    main()
