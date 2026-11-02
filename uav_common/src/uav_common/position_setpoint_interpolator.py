"""Online pure-position setpoint interpolation."""

from __future__ import annotations

import math
import time
from typing import Sequence


class PositionSetpointInterpolator:

    def __init__(
        self,
        *,
        max_speed_m_s: float = 0.5,
        time_constant_s: float = 0.6,
        max_tracking_error_m: float = 0.8,
        max_dt_s: float = 0.2,
    ) -> None:
        self.max_speed_m_s = float(max_speed_m_s)
        self.time_constant_s = float(time_constant_s)
        self.max_tracking_error_m = float(
            max_tracking_error_m
        )
        self.max_dt_s = float(max_dt_s)

        self.position_setpoint: list[float] | None = None
        self.last_time: float | None = None

    def reset(
        self,
        current_position: Sequence[float],
    ) -> None:
        """Reset the interpolator to the current UAV position."""

        self.position_setpoint = self._parse_position(
            current_position
        )
        self.last_time = time.monotonic()

    def update(
        self,
        current_position: Sequence[float],
        target_position: Sequence[float],
        *,
        max_speed_m_s: float | None = None,
    ) -> list[float]:
        """Generate the next intermediate position setpoint."""

        current = self._parse_position(current_position)
        target = self._parse_position(target_position)

        now = time.monotonic()

        if (
            self.position_setpoint is None
            or self.last_time is None
        ):
            self.reset(current)
            return list(self.position_setpoint)

        dt = max(
            0.0,
            min(self.max_dt_s, now - self.last_time),
        )
        self.last_time = now

        if dt <= 0.0:
            return list(self.position_setpoint)

        # 如果虚拟setpoint领先实际飞机太远，暂停推进
        tracking_error = math.sqrt(
            sum(
                (
                    self.position_setpoint[index]
                    - current[index]
                ) ** 2
                for index in range(3)
            )
        )

        if tracking_error > self.max_tracking_error_m:
            return list(self.position_setpoint)

        delta = [
            target[index] - self.position_setpoint[index]
            for index in range(3)
        ]

        distance = math.sqrt(
            sum(value * value for value in delta)
        )

        if distance <= 1e-6:
            self.position_setpoint = list(target)
            return list(self.position_setpoint)

        # 指数插值系数
        alpha = 1.0 - math.exp(
            -dt / self.time_constant_s
        )

        step = [
            alpha * value for value in delta
        ]

        step_distance = math.sqrt(
            sum(value * value for value in step)
        )

        # 限制每个周期中间位置的最大移动距离
        speed_limit = (
            self.max_speed_m_s
            if max_speed_m_s is None
            else float(max_speed_m_s)
        )

        if (
            not math.isfinite(speed_limit)
            or speed_limit <= 0.0
        ):
            raise ValueError(
                "max_speed_m_s must be a positive finite value"
            )

        max_step = speed_limit * dt

        if step_distance > max_step:
            scale = max_step / step_distance
            step = [value * scale for value in step]

        self.position_setpoint = [
            self.position_setpoint[index] + step[index]
            for index in range(3)
        ]

        return list(self.position_setpoint)

    @staticmethod
    def _parse_position(
        position: Sequence[float],
    ) -> list[float]:
        if len(position) != 3:
            raise ValueError(
                "position must contain [x, y, z]"
            )

        parsed = [float(value) for value in position]

        if not all(math.isfinite(value) for value in parsed):
            raise ValueError(
                "position values must be finite"
            )

        return parsed