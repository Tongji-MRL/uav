
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RobotX UAV Pontoon V3.6
Offline Keypoint Refinement Experiment

复用 V3.5：
- 不修改平台分类
- 不修改 V3.4 的分割与颜色簇补救
- 依据实际轮廓改进分支 Tip / Root / Center
- 不能可靠精修时回退 V3.5 原坐标

仅输出图像像素坐标，不输出 XYZ。
本版只用于离线精度测试，不接入飞控。
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from uav_pontoon_v35 import PontoonV35, visualize
from uav_pontoon_v32 import synthetic


# ============================================================
# 1. 查找连续区间
# ============================================================

def bands_from_flags(flags, minimum):

    p = np.asarray(
        flags, dtype=np.uint8
    )

    changes = np.diff(
        np.r_[0, p, 0].astype(np.int16)
    )

    return [
        (int(a), int(b))
        for a, b in zip(
            np.where(changes == 1)[0],
            np.where(changes == -1)[0]
        )
        if b - a >= minimum
    ]


# ============================================================
# 2. 主干边缘直线拟合
# ============================================================

def robust_line(ys, xs):

    ys = np.asarray(
        ys, np.float64
    )

    xs = np.asarray(
        xs, np.float64
    )

    if len(xs) < 10:
        return None

    median = float(
        np.median(xs)
    )

    good = (
        np.abs(xs - median)
        <= max(
            3.,
            0.08 * (
                xs.max() -
                xs.min() + 20
            )
        )
    )

    if np.count_nonzero(good) < 8:
        return None

    return np.polyfit(
        ys[good],
        xs[good],
        1
    )


# ============================================================
# 3. 根据真实掩膜精修关键点
# ============================================================

def refine_keypoints(
    mask_small,
    image_shape,
    geometry,
    old_branches
):

    ys, xs = np.where(
        mask_small > 0
    )

    if len(xs) < 100:
        return None, "empty_mask"

    x0 = int(xs.min())
    y0 = int(ys.min())

    bw = int(xs.max()) - x0 + 1
    bh = int(ys.max()) - y0 + 1

    roi = (
        mask_small[
            y0:y0+bh,
            x0:x0+bw
        ] > 0
    ).astype(np.uint8)

    # 与 V3.5 使用同样的规范化尺寸
    factor = 175. / max(
        bw, bh
    )

    nw = max(
        1, round(bw * factor)
    )

    nh = max(
        1, round(bh * factor)
    )

    roi = cv2.resize(
        roi,
        (nw, nh),
        interpolation=cv2.INTER_NEAREST
    )

    rh, rw = roi.shape

    side = int(
        np.ceil(
            np.hypot(rh, rw)
        )
    ) + 4

    canvas = np.zeros(
        (side, side),
        np.uint8
    )

    px = (side - rw) // 2
    py = (side - rh) // 2

    canvas[
        py:py+rh,
        px:px+rw
    ] = roi

    angle = geometry.get(
        "scan_angle_deg"
    )

    if angle is None:
        return None, "missing_angle"

    rotation = cv2.getRotationMatrix2D(
        (
            (side - 1) / 2,
            (side - 1) / 2
        ),
        float(angle),
        1.
    )

    rotated = cv2.warpAffine(
        canvas,
        rotation,
        (side, side),
        flags=cv2.INTER_NEAREST
    )

    yy, xx = np.where(
        rotated > 0
    )

    if not len(xx):
        return None, "empty_aligned"

    crop_x = int(xx.min())
    crop_y = int(yy.min())

    aligned = rotated[
        crop_y:yy.max()+1,
        crop_x:xx.max()+1
    ]

    flipped = (
        geometry.get(
            "trunk_aligned_side"
        ) == "left"
    )

    if flipped:
        aligned = aligned[:, ::-1]

    h, w = aligned.shape

    if h < 40 or w < 20:
        return None, "too_small"

    # ========================================================
    # 4. 找出四条分支的纵向范围
    # ========================================================

    arm_a = round(.08 * w)
    arm_b = round(.61 * w)

    if arm_b <= arm_a:
        return None, "invalid_arm_region"

    profile = aligned[
        :,
        arm_a:arm_b
    ].mean(axis=1)

    bands = bands_from_flags(
        profile > .48,
        max(
            2,
            round(h * .020)
        )
    )

    if len(bands) != len(old_branches):
        return None, "branch_band_count_mismatch"

    # ========================================================
    # 5. 从分支之间的空隙测量主干左边界
    #
    # 以前：root_x = 0.72 * w
    # 现在：x_left = a*y+b
    # ========================================================

    free = np.ones(
        h, dtype=bool
    )

    for a, b in bands:
        free[
            max(0, a-2):
            min(h, b+2)
        ] = False

    gap_ys = []
    gap_left = []

    for y in range(1, h-1):

        if not free[y]:
            continue

        pxs = np.flatnonzero(
            aligned[y] > 0
        )

        pxs = pxs[
            pxs >= int(.54*w)
        ]

        if len(pxs) < max(
            4, int(.15*w)
        ):
            continue

        left = float(
            np.percentile(
                pxs, 2
            )
        )

        if .56*w <= left <= .82*w:
            gap_ys.append(y)
            gap_left.append(left)

    fit = robust_line(
        gap_ys,
        gap_left
    )

    if fit is None:
        return None, "trunk_edge_not_established"

    # ========================================================
    # 6. 坐标逆变换
    #
    # 规范化方向 ->
    # 旋转前 ->
    # 原图像素坐标
    # ========================================================

    inv = cv2.invertAffineTransform(
        rotation
    )

    sh, sw = mask_small.shape
    oh, ow = image_shape[:2]

    def map_point(x, y):

        if flipped:
            x = w - 1 - x

        q = inv @ np.array(
            [
                x + crop_x,
                y + crop_y,
                1.
            ],
            np.float64
        )

        u = (
            (
                (q[0] - px) / factor
                + x0
            ) * ow / sw
        )

        v = (
            (
                (q[1] - py) / factor
                + y0
            ) * oh / sh
        )

        return [
            round(float(u), 1),
            round(float(v), 1)
        ]

    # ========================================================
    # 7. 逐条计算 Tip / Root / Center
    # ========================================================

    new_arms = []
    displacement = []

    for old, (a, b) in zip(
        old_branches,
        bands
    ):

        ymid = (a+b-1) / 2.

        # 主干左边界与分支所在行的交点
        root_x = float(
            np.polyval(
                fit,
                ymid
            )
        )

        if not (
            .56*w < root_x < .82*w
        ):
            return None, "root_outside_trunk"

        # 搜索完整左侧分支
        # 不再丢掉最左侧8%的区域
        arm_limit = max(
            1,
            min(
                w,
                int(round(root_x-2))
            )
        )

        arm_roi = aligned[
            a:b,
            :arm_limit
        ]

        support = np.count_nonzero(
            arm_roi,
            axis=0
        )

        good_columns = np.where(
            support >= max(
                2,
                int((b-a)*.20)
            )
        )[0]

        if len(good_columns) < max(
            6,
            int(.20*w)
        ):
            return None, "arm_too_fragmented"

        tip_x = float(
            good_columns.min()
        )

        # ====================================================
        # 8. 通过分支像素拟合中心线
        # 减小透视和残余旋转带来的误差
        # ====================================================

        colx = []
        coly = []

        for x in good_columns:

            yvals = np.flatnonzero(
                arm_roi[:, x]
            )

            if len(yvals) < 2:
                continue

            colx.append(
                float(x)
            )

            coly.append(
                float(
                    a + np.median(yvals)
                )
            )

        if (
            len(colx) >= 8
            and np.ptp(colx) >= 8
        ):

            line = np.polyfit(
                colx,
                coly,
                1
            )

            tip_y = float(
                np.polyval(
                    line, tip_x
                )
            )

            root_y = float(
                np.polyval(
                    line, root_x
                )
            )

        else:

            tip_y = float(ymid)
            root_y = float(ymid)

        # 若拟合明显偏离该分支，回退到条带中心
        band_pad = max(
            3.,
            .60*(b-a)
        )

        if (
            abs(tip_y - ymid) > band_pad
            or abs(root_y - ymid) > 1.5*band_pad
        ):

            tip_y = float(ymid)
            root_y = float(ymid)

        tip = map_point(
            tip_x,
            tip_y
        )

        root = map_point(
            root_x,
            root_y
        )

        center = [
            round(
                (tip[0] + root[0]) / 2,
                1
            ),
            round(
                (tip[1] + root[1]) / 2,
                1
            )
        ]

        # ====================================================
        # 9. 几何安全检查
        # ====================================================

        for p in (
            tip, root, center
        ):

            if not (
                0 <= p[0] < ow
                and 0 <= p[1] < oh
            ):

                return None, "point_outside_image"

        delta = float(
            np.linalg.norm(
                np.asarray(root)
                -
                np.asarray(
                    old["root_uv_px"]
                )
            )
        )

        if delta > max(
            35.,
            .15 * max(oh, ow)
        ):

            return None, "excessive_root_shift"

        displacement.append(delta)

        item = dict(old)

        item["tip_uv_px"] = tip
        item["root_uv_px"] = root
        item["center_uv_px"] = center

        item["is_estimated"] = True

        new_arms.append(item)

    # ========================================================
    # 10. 返回精修关键点
    # ========================================================

    return {
        "branches": new_arms,

        "mean_root_shift_px": round(
            float(
                np.mean(displacement)
            ),
            2
        ),

        "measured_trunk_left_fit": [
            round(float(t), 4)
            for t in fit
        ]

    }, "ok"


# ============================================================
# 11. V3.6 主检测器
# ============================================================

class PontoonV36:

    def __init__(
        self,
        max_side=480,
        expected_branches=4
    ):

        self.base = PontoonV35(
            max_side=max_side,
            expected_branches=expected_branches
        )

    def detect(self, image):

        t0 = time.perf_counter()

        info, mask = self.base.detect(
            image
        )

        info = dict(info)

        info["refinement_applied"] = False

        if not (
            info.get("layout_verified")
            and info.get("keypoints_valid")
        ):

            info["refinement_reason"] = (
                "no_verified_layout"
            )

            return info, mask

        refined, reason = refine_keypoints(
            mask,
            image.shape,
            info.get(
                "geometry_details",
                {}
            ),
            info["branches"]
        )

        info["refinement_reason"] = reason

        if refined is not None:

            info.update(refined)

            info["refinement_applied"] = True

        info["processing_ms"] = round(
            (
                time.perf_counter() - t0
            ) * 1000,
            1
        )

        info["xyz_valid"] = False

        return info, mask


# ============================================================
# 12. 主程序
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    modes = parser.add_mutually_exclusive_group(
        required=True
    )

    modes.add_argument(
        "--image"
    )

    modes.add_argument(
        "--self-test",
        action="store_true"
    )

    parser.add_argument(
        "--output"
    )

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

    detector = PontoonV36(
        max_side=args.max_side
    )

    # ========================================================
    # 模式A：基础自测
    # ========================================================

    if args.self_test:

        ok_count = 0

        cases = (
            (
                "blue_0",
                (150, 75, 25),
                0
            ),
            (
                "gray_37",
                (125, 125, 125),
                37
            ),
            (
                "brown_82",
                (80, 105, 130),
                82
            )
        )

        for name, color, angle in cases:

            frame = synthetic(
                color,
                angle
            )

            info, _ = detector.detect(
                frame
            )

            ok = bool(
                info.get("layout_verified")
                and info.get("refinement_applied")
                and len(
                    info.get("branches", [])
                ) == 4
            )

            print(
                name,

                "PASS" if ok else "FAIL",

                "refined=",
                info.get("refinement_applied"),

                "reason=",
                info.get("refinement_reason")
            )

            ok_count += int(ok)

        print(
            f"SELF TEST: {ok_count}/3"
        )

        raise SystemExit(
            0 if ok_count == 3 else 1
        )

    # ========================================================
    # 模式B：静态图像测试
    # ========================================================

    frame = cv2.imread(
        args.image
    )

    if frame is None:

        parser.error(
            "Cannot read image: "
            + args.image
        )

    result, mask = detector.detect(
        frame
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )

    output = (
        args.output
        or str(
            Path(args.image).with_name(
                Path(args.image).stem
                + "_pontoon_v36.jpg"
            )
        )
    )

    if not cv2.imwrite(
        output,
        visualize(
            frame,
            result,
            mask
        )
    ):

        raise RuntimeError(
            "Cannot save " + output
        )

    print(
        "DEBUG_IMAGE:",
        output
    )


if __name__ == "__main__":
    main()
