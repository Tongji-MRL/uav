
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RobotX 2026 UAV Task3 Pontoon V3.4

V3.2 + 相近颜色簇合并补救

原则：
1. 不使用YOLO
2. 不固定浮台颜色
3. 不修改V3.2原代码
4. 仅当FULL_PLATFORM识别出3条分支时二次检查
5. 只有真实候选区域支持4分支时才更新结果
6. 未标定XYZ始终无效
7. 不发送任何飞控指令
"""

import argparse
import json
import time

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from uav_pontoon_v32 import (
    Detector as V32Detector,
    check_full,
    clean,
    draw_debug
)


# ============================================================
# 1. V3.4检测器
# ============================================================

class PontoonV34:

    def __init__(
        self,
        max_side=480,
        k=9,
        max_lab_distance=8.0
    ):

        self.max_side = int(max_side)
        self.k = int(k)

        self.max_lab_distance = float(
            max_lab_distance
        )

        self.baseline = V32Detector(
            max_side=self.max_side,
            k=self.k
        )

    # ========================================================
    # 2. 先执行V3.2
    # ========================================================

    def detect(self, frame):

        t0 = time.perf_counter()

        info, base_mask = (
            self.baseline.detect(frame)
        )

        info = dict(info)

        info["rescue_applied"] = False

        # 非FULL_PLATFORM或不是3分支，
        # 全部直接保留V3.2的原结果。
        if not (
            info.get("status")
            == "FULL_PLATFORM"
            and info.get("branch_count") == 3
        ):

            return info, base_mask

        method = str(
            info.get(
                "candidate_method",
                ""
            )
        )

        # 当前补救只支持Lab单簇候选
        if not method.startswith("lab_"):
            return info, base_mask

        try:

            seed_id = int(
                method.split("_")[1]
            )

        except (IndexError, ValueError):

            return info, base_mask

        if not (
            0 <= seed_id < self.k
        ):
            return info, base_mask

        # ====================================================
        # 3. 重建与V3.2完全一致的Lab聚类
        # ====================================================

        H, W = frame.shape[:2]

        scale = min(
            1.0,
            self.max_side / float(max(W, H))
        )

        if scale < 1.0:

            small = cv2.resize(
                frame,
                (
                    round(W * scale),
                    round(H * scale)
                ),
                interpolation=cv2.INTER_AREA
            )

        else:
            small = frame

        h, w = small.shape[:2]

        lab = cv2.cvtColor(
            cv2.GaussianBlur(
                small,
                (7, 7),
                0
            ),
            cv2.COLOR_BGR2LAB
        )

        cv2.setRNGSeed(5)

        _, labels, centers = cv2.kmeans(
            lab.reshape(
                -1, 3
            ).astype(np.float32),

            self.k,

            None,

            (
                cv2.TERM_CRITERIA_EPS
                +
                cv2.TERM_CRITERIA_MAX_ITER,
                16,
                0.4
            ),

            2,

            cv2.KMEANS_PP_CENTERS
        )

        labels = labels.reshape(h, w)

        old_binary = base_mask > 0

        old_area = int(
            np.count_nonzero(old_binary)
        )

        if (
            old_area < 50
            or old_binary.shape
            != labels.shape
        ):

            return info, base_mask

        # ====================================================
        # 4. 寻找与当前颜色簇相近的其他颜色簇
        # ====================================================

        candidates = []

        for partner in range(self.k):

            if partner == seed_id:
                continue

            distance = float(
                np.linalg.norm(
                    centers[seed_id]
                    -
                    centers[partner]
                )
            )

            if distance > self.max_lab_distance:
                continue

            # 只合并两个颜色簇
            # 避免把大量背景一次性吞入
            combined = clean(
                np.isin(
                    labels,
                    [seed_id, partner]
                )
            )

            n, components, stats, _ = (
                cv2.connectedComponentsWithStats(
                    combined,
                    8
                )
            )

            # =================================================
            # 5. 检查合并之后的连通区域
            # =================================================

            for cid in range(1, n):

                x, y, bw, bh, area = map(
                    int,
                    stats[cid]
                )

                if (
                    area / (h*w) < 0.008
                    or min(bw, bh) < 20
                ):
                    continue

                # 图像边界截断时不进行补救确认
                if min(
                    x,
                    y,
                    w - (x+bw),
                    h - (y+bh)
                ) <= 2:
                    continue

                component_mask = (
                    components == cid
                )

                overlap = int(
                    np.count_nonzero(
                        component_mask
                        &
                        old_binary
                    )
                )

                overlap_ratio = (
                    overlap / max(old_area, 1)
                )

                area_ratio = (
                    area / max(old_area, 1)
                )

                # 必须与旧目标高度重合
                # 不能把附近另一块背景当成补充
                if (
                    overlap_ratio < 0.65
                    or not (
                        1.04 <= area_ratio <= 1.90
                    )
                ):
                    continue

                # =============================================
                # 6. 合并区域重新做完整平台几何验证
                # =============================================

                roi = component_mask[
                    y:y+bh,
                    x:x+bw
                ].astype(np.uint8)

                s = 175.0 / max(roi.shape)

                normalized = cv2.resize(
                    roi,
                    (
                        max(
                            1,
                            round(bw * s)
                        ),
                        max(
                            1,
                            round(bh * s)
                        )
                    ),
                    interpolation=cv2.INTER_NEAREST
                )

                geo = check_full(
                    normalized,
                    step=5
                )

                # 三项硬性约束：
                # 1. 完整平台几何必须通过
                # 2. 必须数出4条分支
                # 3. 几何评分需要实际改善
                if (
                    not geo.get("passed")
                    or geo.get("branches") != 4
                    or geo["score"]
                    < info["geometry_score"] + 0.01
                ):
                    continue

                candidates.append((
                    float(geo["score"]),
                    overlap_ratio,
                    -distance,

                    partner,

                    x, y, bw, bh,

                    component_mask,
                    geo,
                    area_ratio,
                    distance
                ))

        # ====================================================
        # 7. 无有效补救：保留原V3.2结果
        # ====================================================

        if not candidates:

            info["processing_ms"] = round(
                (
                    time.perf_counter() - t0
                ) * 1000,
                1
            )

            return info, base_mask

        # ====================================================
        # 8. 选择最优四分支候选
        # ====================================================

        best = max(
            candidates,
            key=lambda c: (
                c[0],
                c[1],
                c[2]
            )
        )

        (
            score,
            overlap,
            _,
            partner,
            x, y, bw, bh,
            component,
            geo,
            area_ratio,
            distance
        ) = best

        mask = (
            component.astype(np.uint8)
            * 255
        )

        info.update({

            "status": "FULL_PLATFORM",

            "detected": True,

            "branch_count": 4,

            "geometry_score": round(
                score, 4
            ),

            "geometry_details": geo,

            "candidate_method": (
                f"lab_pair_{seed_id}_{partner}"
            ),

            "bbox_xywh_px": [
                round(v / scale)
                for v in (
                    x, y, bw, bh
                )
            ],

            "mask_fraction": round(
                float(
                    mask.mean() / 255.0
                ),
                4
            ),

            "rescue_applied": True,

            "lab_color_distance": round(
                distance, 3
            ),

            "rescue_overlap_fraction": round(
                overlap, 3
            ),

            "rescue_area_ratio": round(
                area_ratio, 3
            ),

            "xyz_valid": False,

            "processing_ms": round(
                (
                    time.perf_counter() - t0
                ) * 1000,
                1
            )
        })

        return info, mask


# ============================================================
# 9. 主程序
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "RobotX UAV Pontoon V3.4"
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

    args = parser.parse_args()

    if (
        args.max_side < 180
        or args.period <= 0
    ):

        parser.error(
            "Require max-side>=180 and period>0"
        )

    detector = PontoonV34(
        max_side=args.max_side
    )

    # ========================================================
    # 10. 静态图片模式
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
                    + "_pontoon_v34.jpg"
                )
            )
        )

        debug = draw_debug(
            frame,
            info,
            mask
        )

        if not cv2.imwrite(
            output,
            debug
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
    # 11. ROS2独立节点
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

    class PontoonNode(Node):

        def __init__(self):

            super().__init__(
                "uav_pontoon_v34_isolated"
            )

            self.sub = (
                self.create_subscription(
                    CompressedImage,
                    args.topic,
                    self.on_image,
                    qos_profile_sensor_data
                )
            )

            self.pub = (
                self.create_publisher(
                    String,
                    "/robotx/perception/uav/pontoon/v34/status",
                    10
                )
            )

            self.dbg = (
                self.create_publisher(
                    CompressedImage,
                    "/robotx/perception/uav/pontoon/v34/debug/compressed",
                    1
                )
            )

            self.history = deque(
                maxlen=5
            )

            self.last = 0.0

            self.get_logger().info(
                "Pontoon V3.4 isolated: "
                + args.topic
            )

        def on_image(self, msg):

            now = time.monotonic()

            if (
                now-self.last
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

                self.get_logger().warning(
                    "Compressed image decode failed"
                )

                return

            info, mask = detector.detect(
                frame
            )

            self.history.append(
                info.get("status")
                == "FULL_PLATFORM"
            )

            info["confirmed_3_of_5"] = (
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

            self.pub.publish(status)

            if (
                self.dbg.get_subscription_count()
                > 0
            ):

                ok, jpg = cv2.imencode(
                    ".jpg",

                    draw_debug(
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

                    debug = CompressedImage()

                    debug.header = msg.header
                    debug.format = "jpeg"
                    debug.data = jpg.tobytes()

                    self.dbg.publish(debug)

    rclpy.init()

    node = PontoonNode()

    try:

        rclpy.spin(node)

    finally:

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
