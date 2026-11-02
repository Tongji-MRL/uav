
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RobotX 2026 UAV Task3 Pontoon V3.5

基于 V3.4 新增完整平台二维几何关键点。

输出：
1. 主干顶部像素坐标
2. 主干底部像素坐标
3. 主干中心与图像内方向
4. 每条分支的端点、根部、中心
5. 布局一致性验证
6. ROS2独立状态和调试图

注意：
- 不使用YOLO
- 不改变V3.4原有分割和识别算法
- B1-B4仅为图像几何顺序
- 坐标为原始图像uv，不是世界XYZ
- 所有关键点是轮廓几何估计值
- 不输出飞控控制指令
"""

import argparse
import json
import math
import time

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from uav_pontoon_v34 import PontoonV34
from uav_pontoon_v32 import draw_debug, synthetic


# ============================================================
# 1. 连续区间查找
# ============================================================

def intervals(flags, minimum):

    d = np.diff(
        np.r_[
            0,
            np.asarray(flags, np.int8),
            0
        ]
    )

    return [
        (int(a), int(b))
        for a, b in zip(
            np.where(d == 1)[0],
            np.where(d == -1)[0]
        )
        if b - a >= minimum
    ]


# ============================================================
# 2. 从检测掩膜提取几何关键点
# ============================================================

def extract_keypoints(
    mask_small,
    original_shape,
    geometry,
    expected
):

    failure = {
        "keypoints_valid": False,
        "keypoint_reason": "unverified"
    }

    if (
        not geometry.get("passed")
        or geometry.get("scan_angle_deg") is None
    ):

        return {
            **failure,
            "keypoint_reason":
                "missing_full_geometry"
        }

    ys, xs = np.where(
        mask_small > 0
    )

    if len(xs) < 100:

        return {
            **failure,
            "keypoint_reason": "empty_mask"
        }

    x0 = int(xs.min())
    y0 = int(ys.min())

    bw = int(xs.max() - x0 + 1)
    bh = int(ys.max() - y0 + 1)

    roi = (
        mask_small[
            y0:y0+bh,
            x0:x0+bw
        ] > 0
    ).astype(np.uint8)

    # 与V3.2几何检查保持相同的归一化尺寸
    factor = 175.0 / max(bw, bh)

    nw = max(
        1,
        round(bw * factor)
    )

    nh = max(
        1,
        round(bh * factor)
    )

    roi = cv2.resize(
        roi,
        (nw, nh),
        interpolation=cv2.INTER_NEAREST
    )

    hr, wr = roi.shape

    # ========================================================
    # 3. 将浮台旋转到标准几何方向
    # ========================================================

    side = int(
        math.ceil(
            math.hypot(hr, wr)
        )
    ) + 4

    canvas = np.zeros(
        (side, side),
        np.uint8
    )

    pad_x = (side - wr) // 2
    pad_y = (side - hr) // 2

    canvas[
        pad_y:pad_y+hr,
        pad_x:pad_x+wr
    ] = roi

    angle = float(
        geometry["scan_angle_deg"]
    )

    matrix = cv2.getRotationMatrix2D(
        (
            (side - 1) / 2,
            (side - 1) / 2
        ),
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        canvas,
        matrix,
        (side, side),
        flags=cv2.INTER_NEAREST
    )

    ry, rx = np.where(
        rotated > 0
    )

    if not len(rx):

        return {
            **failure,
            "keypoint_reason":
                "empty_aligned_mask"
        }

    crop_x = int(rx.min())
    crop_y = int(ry.min())

    aligned = rotated[
        crop_y:ry.max()+1,
        crop_x:rx.max()+1
    ]

    # 与V3.2中check_full的翻转约定一致
    # 最终将长主干统一放到右侧
    flipped = (
        geometry.get(
            "trunk_aligned_side"
        ) == "left"
    )

    if flipped:
        aligned = aligned[:, ::-1]

    h, w = aligned.shape

    # ========================================================
    # 4. 主干和横向分支的相对区域
    # ========================================================

    arm_a = round(.08 * w)
    arm_b = round(.61 * w)

    trunk_a = round(.74 * w)
    trunk_b = round(.96 * w)

    if (
        arm_b <= arm_a
        or trunk_b <= trunk_a
    ):

        return {
            **failure,
            "keypoint_reason":
                "degenerate_alignment"
        }

    # 逐行统计分支区域的像素占有率
    profile = aligned[
        :,
        arm_a:arm_b
    ].mean(axis=1)

    bands = intervals(
        profile > .48,
        max(
            2,
            round(h * .020)
        )
    )

    # 分支计数必须与V3.4的结果一致
    if len(bands) != int(expected):

        return {
            **failure,
            "keypoint_reason":
                "branch_count_disagreement",

            "extracted_branch_count":
                len(bands),

            "reported_branch_count":
                int(expected)
        }

    # ========================================================
    # 5. 坐标反变换
    #
    # 标准方向 ->
    # 旋转前 ->
    # 归一化前 ->
    # 原始图像分辨率
    # ========================================================

    inv = cv2.invertAffineTransform(
        matrix
    )

    sh, sw = mask_small.shape
    oh, ow = original_shape[:2]

    def to_original(x, y):

        # 撤销水平翻转
        unflip_x = (
            w - 1 - x
            if flipped
            else x
        )

        # 恢复到旋转后完整画布
        p = np.array([
            unflip_x + crop_x,
            y + crop_y,
            1.0
        ])

        # 逆仿射变换
        p0 = inv @ p

        # 撤销填充和归一化
        small_x = (
            (p0[0] - pad_x) / factor
            + x0
        )

        small_y = (
            (p0[1] - pad_y) / factor
            + y0
        )

        # 恢复原始图像分辨率
        return [
            float(
                round(
                    float(
                        small_x * ow / sw
                    ),
                    1
                )
            ),
            float(
                round(
                    float(
                        small_y * oh / sh
                    ),
                    1
                )
            )
        ]

    # ========================================================
    # 6. 提取长主干两端
    # ========================================================

    trunk = aligned[
        :,
        trunk_a:trunk_b
    ]

    trunk_rows = np.where(
        trunk.mean(axis=1) > .52
    )[0]

    if len(trunk_rows) < 10:

        return {
            **failure,
            "keypoint_reason":
                "trunk_not_continuous"
        }

    top_y = int(trunk_rows[0])
    bottom_y = int(trunk_rows[-1])

    def trunk_x(y):

        lo = max(0, y - 3)
        hi = min(h, y + 4)

        local_cols = np.where(
            aligned[
                lo:hi,
                trunk_a:trunk_b
            ] > 0
        )[1]

        if len(local_cols):

            return float(
                trunk_a
                + np.median(local_cols)
            )

        return float(
            (trunk_a + trunk_b) * .5
        )

    top = to_original(
        trunk_x(top_y),
        top_y
    )

    bottom = to_original(
        trunk_x(bottom_y),
        bottom_y
    )

    trunk_center = [
        round(
            (top[0] + bottom[0]) / 2,
            1
        ),
        round(
            (top[1] + bottom[1]) / 2,
            1
        )
    ]

    # ========================================================
    # 7. 提取所有横向分支
    # ========================================================

    branches = []

    for index, (start, end) in enumerate(
        bands, 1
    ):

        band = aligned[
            start:end,
            arm_a:arm_b
        ]

        _, columns = np.where(
            band > 0
        )

        if not len(columns):

            return {
                **failure,
                "keypoint_reason":
                    "empty_branch_band"
            }

        # 采用靠外侧像素的低百分位位置
        # 避免单个噪声点过度影响端点
        tip_x = float(
            arm_a
            + np.percentile(
                columns, 4
            )
        )

        center_y = float(
            (start + end - 1) / 2
        )

        # 根部位置是靠近主干的几何估计
        root_x = float(
            .72 * w
        )

        branches.append({

            "id": f"B{index}",

            "tip_uv_px": to_original(
                tip_x,
                center_y
            ),

            "root_uv_px": to_original(
                root_x,
                center_y
            ),

            "center_uv_px": to_original(
                (tip_x + root_x) / 2,
                center_y
            ),

            "is_estimated": True
        })

    # ========================================================
    # 8. 检查关键点是否在图像范围内
    # ========================================================

    all_points = [
        top,
        bottom
    ]

    for branch in branches:

        all_points.extend([
            branch["tip_uv_px"],
            branch["root_uv_px"],
            branch["center_uv_px"]
        ])

    inside = all(
        0 <= p[0] < ow
        and 0 <= p[1] < oh
        for p in all_points
    )

    if not inside:

        return {
            **failure,
            "keypoint_reason":
                "keypoint_out_of_image"
        }

    dx = bottom[0] - top[0]
    dy = bottom[1] - top[1]

    image_angle = math.degrees(
        math.atan2(dy, dx)
    )

    return {

        "keypoints_valid": True,

        "keypoint_reason": "ok",

        "pixel_coordinate_system":
            "original_image_uv_x_right_y_down",

        "trunk_top_uv_px": top,

        "trunk_bottom_uv_px": bottom,

        "trunk_center_uv_px":
            trunk_center,

        "trunk_angle_image_deg":
            round(image_angle, 2),

        "branches": branches,

        "extracted_branch_count":
            len(branches),

        "branch_ordering":
            "aligned_image_top_to_bottom_not_semantic_ID"
    }


# ============================================================
# 9. V3.5主检测器
# ============================================================

class PontoonV35:

    def __init__(
        self,
        max_side=480,
        expected_branches=4
    ):

        self.base = PontoonV34(
            max_side=max_side
        )

        self.expected_branches = int(
            expected_branches
        )

    def detect(self, frame):

        info, mask = self.base.detect(
            frame
        )

        info = dict(info)

        info["keypoints_valid"] = False
        info["layout_verified"] = False

        info["expected_branch_count"] = (
            self.expected_branches
        )

        info["xyz_valid"] = False

        # 未找到完整平台时，不输出虚假的关键点
        if (
            info.get("status")
            != "FULL_PLATFORM"
        ):

            info["keypoint_reason"] = (
                "not_a_confirmed_full_platform"
            )

            return info, mask

        geometry = (
            info.get("geometry_details")
            or {}
        )

        keypoints = extract_keypoints(
            mask,
            frame.shape,
            geometry,
            int(
                info.get(
                    "branch_count", 0
                )
            )
        )

        info.update(keypoints)

        # 布局校验严格要求：
        # 原检测分支数 == 关键点数 == 预期分支数
        info["layout_verified"] = bool(

            info["keypoints_valid"]

            and info.get(
                "branch_count"
            ) == self.expected_branches

            and info.get(
                "extracted_branch_count"
            ) == self.expected_branches
        )

        return info, mask


# ============================================================
# 10. 关键点可视化
# ============================================================

def visualize(frame, info, mask):

    out = draw_debug(
        frame,
        info,
        mask
    )

    if not info.get(
        "keypoints_valid"
    ):
        return out

    thickness = max(
        2,
        frame.shape[1] // 800
    )

    radius = max(
        5,
        frame.shape[1] // 320
    )

    def pt(p):
        return tuple(
            int(round(x))
            for x in p
        )

    # ========================================================
    # 10.1 长主干轴线
    # ========================================================

    top = pt(
        info["trunk_top_uv_px"]
    )

    bottom = pt(
        info["trunk_bottom_uv_px"]
    )

    cv2.line(
        out,
        top,
        bottom,
        (255, 255, 0),
        thickness,
        cv2.LINE_AA
    )

    for p in (
        top,
        bottom
    ):

        cv2.circle(
            out,
            p,
            radius,
            (255, 255, 0),
            -1,
            cv2.LINE_AA
        )

    # ========================================================
    # 10.2 每条横向分支
    # ========================================================

    for arm in info["branches"]:

        root = pt(
            arm["root_uv_px"]
        )

        tip = pt(
            arm["tip_uv_px"]
        )

        center = pt(
            arm["center_uv_px"]
        )

        cv2.line(
            out,
            root,
            tip,
            (255, 0, 255),
            thickness,
            cv2.LINE_AA
        )

        cv2.circle(
            out,
            center,
            radius,
            (0, 255, 255),
            -1,
            cv2.LINE_AA
        )

        text_pos = (
            center[0] + radius + 2,
            center[1] - radius - 2
        )

        font_scale = max(
            .55,
            frame.shape[1] / 2800
        )

        # 黑色文字描边
        cv2.putText(
            out,
            arm["id"],
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            max(2, thickness + 2),
            cv2.LINE_AA
        )

        # 白色文字
        cv2.putText(
            out,
            arm["id"],
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness // 2),
            cv2.LINE_AA
        )

    return out


# ============================================================
# 11. 程序入口
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "RobotX UAV Pontoon V3.5 2D Keypoints"
        )
    )

    mode = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    mode.add_argument(
        "--image"
    )

    mode.add_argument(
        "--self-test",
        action="store_true"
    )

    mode.add_argument(
        "--ros",
        action="store_true"
    )

    parser.add_argument(
        "--output"
    )

    parser.add_argument(
        "--topic",
        default="/oak/rgb/image_raw/compressed"
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=480
    )

    parser.add_argument(
        "--period",
        type=float,
        default=1.5
    )

    parser.add_argument(
        "--expected-branches",
        type=int,
        default=4
    )

    args = parser.parse_args()

    if (
        args.max_side < 180
        or args.expected_branches < 3
        or args.period <= 0
    ):

        parser.error(
            "Require max-side>=180, "
            "expected-branches>=3, period>0"
        )

    detector = PontoonV35(
        max_side=args.max_side,
        expected_branches=args.expected_branches
    )

    # ========================================================
    # 12. 内置合成测试
    # ========================================================

    if args.self_test:

        correct = 0

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
                info["layout_verified"]
            )

            correct += int(ok)

            print(
                name,
                "PASS" if ok else "FAIL",
                "status=",
                info["status"],
                "arms=",
                info.get(
                    "extracted_branch_count",
                    0
                ),
                "reason=",
                info.get(
                    "keypoint_reason"
                )
            )

        print(
            f"KEYPOINT SELF TEST: {correct}/3"
        )

        raise SystemExit(
            0 if correct == 3 else 1
        )

    # ========================================================
    # 13. 单张图片模式
    # ========================================================

    if args.image:

        frame = cv2.imread(
            args.image
        )

        if frame is None:

            parser.error(
                "Cannot read image: "
                + args.image
            )

        info, mask = detector.detect(
            frame
        )

        print(
            json.dumps(
                info,
                ensure_ascii=False,
                indent=2
            )
        )

        output = (
            args.output
            or str(
                Path(args.image).with_name(
                    Path(args.image).stem
                    + "_pontoon_v35.jpg"
                )
            )
        )

        if not cv2.imwrite(
            output,
            visualize(
                frame,
                info,
                mask
            )
        ):

            raise RuntimeError(
                "Cannot save: " + output
            )

        print(
            "DEBUG_IMAGE:",
            output
        )

        return

    # ========================================================
    # 14. ROS2独立测试节点
    # ========================================================

    import rclpy

    from rclpy.node import Node

    from rclpy.qos import (
        qos_profile_sensor_data
    )

    from sensor_msgs.msg import (
        CompressedImage
    )

    from std_msgs.msg import String

    class NodeV35(Node):

        def __init__(self):

            super().__init__(
                "uav_pontoon_v35_isolated"
            )

            self.sub = (
                self.create_subscription(
                    CompressedImage,
                    args.topic,
                    self.callback,
                    qos_profile_sensor_data
                )
            )

            self.state_pub = (
                self.create_publisher(
                    String,
                    "/robotx/perception/uav/pontoon/v35/status",
                    10
                )
            )

            self.debug_pub = (
                self.create_publisher(
                    CompressedImage,
                    "/robotx/perception/uav/pontoon/v35/debug/compressed",
                    1
                )
            )

            self.history = deque(
                maxlen=5
            )

            self.last = 0.0

            self.get_logger().info(
                "V3.5 keypoints test on "
                + args.topic
            )

        def callback(self, msg):

            now = time.monotonic()

            if (
                now - self.last
                < args.period
            ):
                return

            self.last = now

            frame = cv2.imdecode(
                np.frombuffer(
                    msg.data,
                    np.uint8
                ),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                return

            info, mask = detector.detect(
                frame
            )

            self.history.append(
                bool(
                    info["layout_verified"]
                )
            )

            # 只是最近5帧的布局出现统计
            # 没有关联目标ID，不能用于自动控制
            info[
                "layout_seen_3_of_5_untracked"
            ] = (
                len(self.history) == 5
                and sum(self.history) >= 3
            )

            info["frame_id"] = (
                msg.header.frame_id
            )

            info["stamp"] = {
                "sec": int(
                    msg.header.stamp.sec
                ),
                "nanosec": int(
                    msg.header.stamp.nanosec
                )
            }

            status = String()

            status.data = json.dumps(
                info,
                ensure_ascii=False
            )

            self.state_pub.publish(
                status
            )

            if (
                self.debug_pub
                .get_subscription_count()
                > 0
            ):

                ok, jpg = cv2.imencode(
                    ".jpg",
                    visualize(
                        frame,
                        info,
                        mask
                    ),
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        75
                    ]
                )

                if ok:

                    out = CompressedImage()

                    out.header = msg.header
                    out.format = "jpeg"
                    out.data = jpg.tobytes()

                    self.debug_pub.publish(
                        out
                    )

    rclpy.init()

    node = NodeV35()

    try:
        rclpy.spin(node)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
