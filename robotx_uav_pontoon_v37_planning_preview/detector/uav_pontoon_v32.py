
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RobotX 2026 - UAV Task3 Pontoon Detector V3.2

主要功能：
1. OpenCV 非YOLO识别
2. Lab自适应颜色分割，保留V2的完整轮廓能力
3. 多种候选区域搜索，不固定浮台颜色
4. 长主干、多分支结构验证
5. 修复V3.1的0/255几何计算错误
6. 严格检查U形两条长臂
7. 支持静态图片、自测和ROS2输入
8. 独立ROS2输出，不连接飞控

重要：
geometry_score不是概率。
PARTIAL_U_CANDIDATE不是完整比赛浮台。
未完成标定时xyz_valid始终为False。
"""

import argparse
import cv2
import numpy as np
import time
import json

from pathlib import Path


# =========================================================
# 1. 查找连续区间
# =========================================================

def intervals(flags, min_len):

    d = np.diff(
        np.r_[
            0,
            np.asarray(flags, dtype=np.int8),
            0
        ]
    )

    return [
        (int(a), int(b))
        for a, b in zip(
            np.where(d == 1)[0],
            np.where(d == -1)[0]
        )
        if b - a >= min_len
    ]


# =========================================================
# 2. 二值图旋转
# =========================================================

def rotate_tight(mask, angle):

    h, w = mask.shape

    s = int(
        np.ceil(np.hypot(h, w))
    ) + 4

    canvas = np.zeros(
        (s, s),
        np.uint8
    )

    y = (s - h) // 2
    x = (s - w) // 2

    canvas[y:y+h, x:x+w] = mask

    matrix = cv2.getRotationMatrix2D(
        ((s - 1) / 2, (s - 1) / 2),
        angle,
        1
    )

    rotated = cv2.warpAffine(
        canvas,
        matrix,
        (s, s),
        flags=cv2.INTER_NEAREST
    )

    yy, xx = np.where(
        rotated > 0
    )

    if len(xx) == 0:
        return None

    return rotated[
        yy.min():yy.max()+1,
        xx.min():xx.max()+1
    ]


# =========================================================
# 3. 完整多分支浮台检测
#
# 注意：进入几何计算前必须统一成0/1
# =========================================================

def check_full(mask, step=5):

    mask = (
        mask > 0
    ).astype(np.uint8)

    best = {
        "passed": False,
        "score": 0.0,
        "branches": 0,
        "scan_angle_deg": None
    }

    if (
        min(mask.shape) < 12
        or mask.sum() < 120
    ):
        return best

    for angle in range(
        0, 180, step
    ):

        rotated = rotate_tight(
            mask,
            angle
        )

        if rotated is None:
            continue

        H, W = rotated.shape

        if (
            W < 18
            or H < 1.35 * W
        ):
            continue

        # 允许主干出现在两侧
        for flip in (
            False,
            True
        ):

            m = (
                rotated[:, ::-1]
                if flip
                else rotated
            )

            trunk = m[
                :,
                round(.74*W):round(.96*W)
            ]

            arms = m[
                :,
                round(.08*W):round(.61*W)
            ]

            if (
                not trunk.size
                or not arms.size
            ):
                continue

            trunk_fraction = (
                trunk.mean(axis=1)
            )

            trunk_ok = float(
                (
                    trunk_fraction > .52
                ).mean()
            )

            arm_profile = (
                arms.mean(axis=1)
            )

            bands = intervals(
                arm_profile > .48,
                max(
                    2,
                    round(H*.020)
                )
            )

            count = len(bands)

            if not 3 <= count <= 6:
                continue

            centers = np.array([
                (i+j)*.5
                for i, j in bands
            ])

            gaps = np.diff(centers)

            spacing_cv = float(
                np.std(gaps)
                / max(
                    np.mean(gaps),
                    1.
                )
            )

            gap_fraction = float(
                np.mean(
                    arm_profile < .20
                )
            )

            widths = np.array([
                (j-i)/H
                for i, j in bands
            ])

            mean_width = float(
                widths.mean()
            )

            width_cv = float(
                widths.std()
                / max(
                    mean_width,
                    1e-6
                )
            )

            occupancy = float(
                arms.mean()
            )

            spacing = float(
                np.mean(gaps)/H
            )

            # 规则评分，不是置信概率
            score = (
                .32 * min(
                    1.,
                    trunk_ok/.90
                )
                +
                .25 * min(
                    1.,
                    count/4
                )
                +
                .16 * max(
                    0.,
                    1.-spacing_cv/.4
                )
                +
                .10 * float(
                    .018 <= mean_width <= .20
                )
                +
                .09 * float(
                    .07 <= occupancy <= .48
                )
                +
                .08 * min(
                    1.,
                    gap_fraction/.35
                )
            )

            passed = bool(
                trunk_ok > .84
                and 3 <= count <= 6
                and spacing_cv < .30
                and width_cv < .40
                and .065 < spacing < .45
                and .018 < mean_width < .20
                and .07 < occupancy < .48
                and gap_fraction > .27
                and score >= .81
            )

            if (
                int(passed),
                score
            ) > (
                int(best["passed"]),
                best["score"]
            ):

                best = {
                    "passed": passed,

                    "score": round(
                        float(score),
                        4
                    ),

                    "branches": count,

                    "scan_angle_deg": angle,

                    "trunk_aligned_side": (
                        "left"
                        if flip
                        else "right"
                    ),

                    "trunk_continuity": round(
                        trunk_ok,
                        3
                    ),

                    "spacing_cv": round(
                        spacing_cv,
                        3
                    ),

                    "arm_gap_fraction": round(
                        gap_fraction,
                        3
                    )
                }

    return best


# =========================================================
# 4. 局部U形检测
#
# 重点修复：
# 1. 所有mask统一为0/1
# 2. 两条长臂在中下部必须同时存在
# 3. 中央空隙必须持续存在
# =========================================================

def check_partial(mask):

    mask = (
        mask > 0
    ).astype(np.uint8)

    best = {
        "passed": False,
        "score": 0.0,
        "bridge_density": 0.0,
        "void_density": 0.0
    }

    for angle in range(
        0, 180, 15
    ):

        rotated = rotate_tight(
            mask,
            angle
        )

        if rotated is None:
            continue

        for flip in (
            False,
            True
        ):

            r = (
                rotated[::-1]
                if flip
                else rotated
            )

            H, W = r.shape

            if (
                min(H, W) < 55
                or not (.55 < W/H < 3.8)
            ):
                continue

            top = r[
                round(H*.08):round(H*.36)
            ]

            mid = r[
                round(H*.55):round(H*.75)
            ]

            lower = r[
                round(H*.78):round(H*.92)
            ]

            if (
                not top.size
                or not mid.size
                or not lower.size
            ):
                continue

            profile = cv2.blur(
                mid.astype(np.float32),
                (13, 1)
            ).mean(axis=0)

            stripes = intervals(
                profile > .24,
                max(
                    8,
                    round(W*.045)
                )
            )

            for i in range(
                len(stripes)
            ):

                for j in range(
                    i+1,
                    len(stripes)
                ):

                    left = stripes[i]
                    right = stripes[j]

                    u = left[1]
                    v = right[0]

                    if v <= u:
                        continue

                    gap = (v-u)/W

                    span = (
                        right[1] - left[0]
                    )/W

                    if (
                        gap < .14
                        or span < .55
                    ):
                        continue

                    bridge = float(
                        top[:, u:v].mean()
                    )

                    empty = float(
                        mid[:, u:v].mean()
                    )

                    empty_lower = float(
                        lower[:, u:v].mean()
                    )

                    # 检查左右两条长臂
                    left_mid = float(
                        mid[
                            :,
                            left[0]:left[1]
                        ].mean()
                    )

                    right_mid = float(
                        mid[
                            :,
                            right[0]:right[1]
                        ].mean()
                    )

                    left_lower = float(
                        lower[
                            :,
                            left[0]:left[1]
                        ].mean()
                    )

                    right_lower = float(
                        lower[
                            :,
                            right[0]:right[1]
                        ].mean()
                    )

                    score = (
                        .35 * min(
                            1.,
                            gap/.32
                        )
                        +
                        .45 * min(
                            1.,
                            bridge/.55
                        )
                        +
                        .20 * (
                            1. - min(
                                1.,
                                empty/.20
                            )
                        )
                    )

                    passed = bool(
                        bridge > .28
                        and empty < .14
                        and empty_lower < .14
                        and bridge-empty > .25
                        and left_mid > .50
                        and right_mid > .50
                        and left_lower > .22
                        and right_lower > .22
                        and score > .70
                    )

                    if (
                        int(passed),
                        score
                    ) > (
                        int(best["passed"]),
                        best["score"]
                    ):

                        best = {
                            "passed": passed,

                            "score": round(
                                score, 4
                            ),

                            "bridge_density": round(
                                bridge, 4
                            ),

                            "void_density": round(
                                empty, 4
                            ),

                            "angle_scan_deg": angle
                        }

    return best


# =========================================================
# 5. 清理候选掩膜
# =========================================================

def clean(mask):

    mask = (
        (mask > 0).astype(np.uint8)
        * 255
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9)
        )
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )
    )

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            8
        )
    )

    output = np.zeros_like(
        mask
    )

    for i in range(1, n):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= max(
            160,
            int(.0035 * mask.size)
        ):

            output[
                labels == i
            ] = 255

    return output


# =========================================================
# 6. 提取外接框
# =========================================================

def bbox(mask):

    yy, xx = np.where(
        mask > 0
    )

    if not len(xx):
        return None

    return (
        int(xx.min()),
        int(yy.min()),

        int(
            xx.max()-xx.min()+1
        ),

        int(
            yy.max()-yy.min()+1
        )
    )


# =========================================================
# 7. 检测器
# =========================================================

class Detector:

    def __init__(
        self,
        max_side=480,
        k=9
    ):

        self.max_side = max_side
        self.k = k

    def detect(self, image):

        start = time.perf_counter()

        H, W = image.shape[:2]

        scale = min(
            1.,
            self.max_side/max(H, W)
        )

        w = round(W*scale)
        h = round(H*scale)

        if scale < 1:

            small = cv2.resize(
                image,
                (w, h),
                interpolation=cv2.INTER_AREA
            )

        else:

            small = image

        # -------------------------------------------------
        # Lab自适应聚类
        # -------------------------------------------------

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
                .4
            ),

            2,

            cv2.KMEANS_PP_CENTERS
        )

        labels = labels.reshape(
            h, w
        )

        chroma = np.linalg.norm(
            centers[:, 1:] - 128,
            axis=1
        )

        index = int(
            chroma.argmax()
        )

        reference = centers[index]

        # -------------------------------------------------
        # 保留V2主要分割策略
        # -------------------------------------------------

        selected = []

        for j, color in enumerate(
            centers
        ):

            cosine = float(
                np.dot(
                    reference[1:] - 128,
                    color[1:] - 128
                )
                /
                max(
                    1.,
                    chroma[index]*chroma[j]
                )
            )

            distance = float(
                np.linalg.norm(
                    reference-color
                )
            )

            if (
                j == index
                or (
                    cosine > .55
                    and distance < 95
                    and chroma[j] > 18
                )
            ):

                selected.append(j)

        primary = clean(
            np.isin(
                labels,
                selected
            )
        )

        # -------------------------------------------------
        # 多候选颜色搜索
        # -------------------------------------------------

        proposals = [
            ("v2_primary", primary)
        ]

        for j in range(self.k):

            proposals.append((
                f"lab_{j}",
                clean(labels == j)
            ))

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2GRAY
        )

        _, thresholded = cv2.threshold(
            cv2.GaussianBlur(
                gray,
                (5, 5),
                0
            ),
            0,
            255,
            cv2.THRESH_BINARY
            +
            cv2.THRESH_OTSU
        )

        proposals += [
            (
                "gray_light",
                clean(thresholded)
            ),
            (
                "gray_dark",
                clean(255-thresholded)
            )
        ]

        # -------------------------------------------------
        # 仅完整平台几何可以优先选择其他候选
        # -------------------------------------------------

        hits = []

        for name, proposal in proposals:

            fraction = float(
                np.count_nonzero(proposal)
                / proposal.size
            )

            if not (
                .009 < fraction < .65
            ):
                continue

            n, comps, stats, _ = (
                cv2.connectedComponentsWithStats(
                    proposal,
                    8
                )
            )

            choices = sorted(
                range(1, n),

                key=lambda i: stats[
                    i,
                    cv2.CC_STAT_AREA
                ],

                reverse=True
            )[:3]

            for component_id in choices:

                x, y, bw, bh, area = map(
                    int,
                    stats[component_id]
                )

                area_fraction = area/(h*w)

                if (
                    area_fraction < .008
                    or min(bw, bh) < 20
                ):
                    continue

                # 图像边界被截断时不确认完整平台
                if min(
                    x,
                    y,
                    w-(x+bw),
                    h-(y+bh)
                ) <= 2:
                    continue

                component = (
                    comps[
                        y:y+bh,
                        x:x+bw
                    ] == component_id
                ).astype(np.uint8)

                resize_factor = (
                    175/max(component.shape)
                )

                resized = cv2.resize(
                    component,
                    (
                        max(
                            1,
                            round(
                                bw*resize_factor
                            )
                        ),
                        max(
                            1,
                            round(
                                bh*resize_factor
                            )
                        )
                    ),
                    interpolation=cv2.INTER_NEAREST
                )

                geometry = check_full(
                    resized
                )

                if geometry["passed"]:

                    component_mask = (
                        np.uint8(
                            comps == component_id
                        )*255
                    )

                    hits.append((
                        geometry["score"],
                        name,
                        component_mask,
                        geometry
                    ))

        # -------------------------------------------------
        # 选择最终结果
        # -------------------------------------------------

        if hits:

            hits.sort(
                key=lambda item: item[0],
                reverse=True
            )

            _, source, chosen, geometry = (
                hits[0]
            )

            status = "FULL_PLATFORM"

        else:

            # 局部U形只使用V2主分割结果
            # 不允许草坪候选凭高分抢占目标
            chosen = primary
            source = "v2_primary"

            geometry = check_partial(
                primary
            )

            primary_fraction = float(
                np.count_nonzero(primary)
                / primary.size
            )

            if (
                geometry["passed"]
                and .025 < primary_fraction < .50
            ):

                status = (
                    "PARTIAL_U_CANDIDATE"
                )

            elif (
                .012 < primary_fraction < .55
            ):

                status = (
                    "SEGMENTED_CANDIDATE"
                )

            else:

                status = "UNKNOWN"

                chosen = np.zeros_like(
                    primary
                )

        # -------------------------------------------------
        # 输出二维坐标
        # -------------------------------------------------

        box = bbox(chosen)

        if box is not None:

            output_bbox = [
                round(value/scale)
                for value in box
            ]

        else:

            output_bbox = None

        result = {
            "status": status,

            "detected": (
                status == "FULL_PLATFORM"
            ),

            "geometry_score":
                geometry["score"],

            "score_is_probability": False,

            "branch_count": (
                geometry.get(
                    "branches",
                    0
                )
                if status == "FULL_PLATFORM"
                else 0
            ),

            "candidate_method": source,

            "bbox_xywh_px": output_bbox,

            "mask_fraction": round(
                float(
                    np.count_nonzero(chosen)
                    / chosen.size
                ),
                4
            ),

            "geometry_details": geometry,

            "xyz_valid": False,

            "processing_ms": round(
                (
                    time.perf_counter()-start
                )*1000,
                1
            )
        }

        return result, chosen


# =========================================================
# 8. 调试图绘制
# =========================================================

def draw_debug(frame, info, small_mask):

    out = frame.copy()

    H, W = frame.shape[:2]

    mask = cv2.resize(
        small_mask,
        (W, H),
        interpolation=cv2.INTER_NEAREST
    )

    color = (
        (50, 230, 50)
        if info["detected"]
        else (0, 185, 255)
    )

    present = mask > 0

    out[present] = (
        .62*out[present]
        +
        .38*np.asarray(color)
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        out,
        contours,
        -1,
        color,
        max(2, W//900)
    )

    box = info[
        "bbox_xywh_px"
    ]

    if box is not None:

        x, y, w, h = box

        cv2.rectangle(
            out,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

    label = (
        f"{info['status']}"
        f" | geom={info['geometry_score']:.3f}"
        f" | branches={info['branch_count']}"
        f" | XYZ INVALID"
    )

    cv2.rectangle(
        out,
        (0, 0),
        (min(W-1, 1100), 44),
        (18, 18, 18),
        -1
    )

    cv2.putText(
        out,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        .7,
        color,
        2,
        cv2.LINE_AA
    )

    return out


# =========================================================
# 9. 合成完整平台
# =========================================================

def synthetic(color, angle):

    image = np.full(
        (480, 640, 3),
        (95, 125, 150),
        np.uint8
    )

    mask = np.zeros(
        (480, 640),
        np.uint8
    )

    for row in range(15):

        for col in range(6):

            occupied = (
                col >= 4
                or row in (1, 5, 9, 13)
            )

            if not occupied:
                continue

            x = 210 + 17*col
            y = 95 + 17*row

            cv2.rectangle(
                mask,
                (x, y),
                (x+15, y+15),
                255,
                -1
            )

    matrix = cv2.getRotationMatrix2D(
        (260, 222),
        angle,
        1
    )

    mask = cv2.warpAffine(
        mask,
        matrix,
        (640, 480)
    )

    image[mask > 0] = color

    return image


# =========================================================
# 10. 内置自测
# =========================================================

def self_test(detector):

    cases = [
        (
            "blue_0",
            synthetic(
                (150, 75, 25), 0
            ),
            True
        ),
        (
            "gray_37",
            synthetic(
                (125, 125, 125), 37
            ),
            True
        ),
        (
            "brown_82",
            synthetic(
                (80, 105, 130), 82
            ),
            True
        ),
        (
            "blank",
            np.full(
                (480, 640, 3),
                (180, 120, 80),
                np.uint8
            ),
            False
        )
    ]

    grid = np.full(
        (480, 640, 3),
        (180, 120, 80),
        np.uint8
    )

    for row in range(12):

        for col in range(9):

            x = 120 + 30*col
            y = 60 + 29*row

            cv2.rectangle(
                grid,
                (x, y),
                (x+26, y+25),
                (100, 90, 100),
                -1
            )

    cases.append((
        "ordinary_grid",
        grid,
        False
    ))

    success = 0

    for name, image, positive in cases:

        result, _ = detector.detect(
            image
        )

        hit = (
            result["status"]
            == "FULL_PLATFORM"
        )

        ok = (
            hit == positive
            and (
                not positive
                or result["branch_count"] == 4
            )
        )

        success += int(ok)

        print(
            name,
            "PASS" if ok else "FAIL",
            result["status"],
            "branches=",
            result["branch_count"],
            "ms=",
            result["processing_ms"]
        )

    print(
        f"{success}/{len(cases)} PASS"
    )

    return success == len(cases)


# =========================================================
# 11. ROS2独立实验节点
# =========================================================

def run_ros(detector, args):

    import rclpy

    from rclpy.node import Node

    from rclpy.qos import (
        qos_profile_sensor_data
    )

    from sensor_msgs.msg import (
        CompressedImage
    )

    from std_msgs.msg import String

    from collections import deque

    class PontoonNode(Node):

        def __init__(self):

            super().__init__(
                "uav_pontoon_v32_isolated"
            )

            self.subscription = (
                self.create_subscription(
                    CompressedImage,
                    args.topic,
                    self.callback,
                    qos_profile_sensor_data
                )
            )

            self.status_pub = (
                self.create_publisher(
                    String,
                    "/robotx/perception/uav/pontoon/v32/status",
                    10
                )
            )

            self.debug_pub = (
                self.create_publisher(
                    CompressedImage,
                    "/robotx/perception/uav/pontoon/v32/debug/compressed",
                    1
                )
            )

            self.history = deque(
                maxlen=5
            )

            self.last = 0.

            self.get_logger().info(
                "V3.2 isolated, image topic: "
                + args.topic
            )

        def callback(self, msg):

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
                    "Cannot decode compressed image"
                )

                return

            info, mask = detector.detect(
                frame
            )

            self.history.append(
                info["detected"]
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

            self.status_pub.publish(
                status
            )

            # 有订阅者时才编码调试图片
            if (
                self.debug_pub.get_subscription_count()
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

                    self.debug_pub.publish(
                        debug
                    )

    rclpy.init()

    node = PontoonNode()

    try:

        rclpy.spin(node)

    finally:

        node.destroy_node()
        rclpy.shutdown()


# =========================================================
# 12. 主入口
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "RobotX UAV pontoon OpenCV v3.2"
        )
    )

    mode = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    mode.add_argument(
        "--image",
        type=str
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
        "--output",
        type=str
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
            "Require --max-side>=180 and --period>0"
        )

    detector = Detector(
        max_side=args.max_side
    )

    if args.self_test:

        passed = self_test(
            detector
        )

        raise SystemExit(
            0 if passed else 1
        )

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
                    + "_pontoon_v32.jpg"
                )
            )
        )

        ok = cv2.imwrite(
            output,
            draw_debug(
                frame,
                info,
                mask
            )
        )

        if not ok:

            raise RuntimeError(
                "Cannot save image: "
                + output
            )

        print(
            "DEBUG_IMAGE:",
            output
        )

        return

    run_ros(
        detector,
        args
    )


if __name__ == "__main__":
    main()
