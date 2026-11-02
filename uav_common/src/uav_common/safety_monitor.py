"""Plain-Python PX4 safety evaluation and detailed reporting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SafetyResult:
    """One safety decision produced from the latest cached telemetry."""

    safe: bool
    reason: str


@dataclass(frozen=True)
class SafetyCheck:
    """One named check included in a detailed safety report."""

    name: str
    passed: bool
    detail: str
    failure_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


class SafetyMonitor:
    """Cache vehicle data and evaluate it without owning ROS entities."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.input_timeout_s = float(config.get("input_timeout_s", 0.5))
        self.minimum_battery_remaining = float(
            config.get("minimum_battery_remaining", 0.20)
        )
        self.battery_timeout_s = float(config.get("battery_timeout_s", 2.0))
        self.require_battery = bool(config.get("require_battery", True))
        self.allow_dead_reckoning = bool(
            config.get("allow_dead_reckoning", False)
        )
        if not math.isfinite(self.input_timeout_s) or self.input_timeout_s <= 0.0:
            raise ValueError("safety.input_timeout_s must be positive")
        if not 0.0 <= self.minimum_battery_remaining <= 1.0:
            raise ValueError(
                "safety.minimum_battery_remaining must be in [0, 1]"
            )
        if not math.isfinite(self.battery_timeout_s) or self.battery_timeout_s <= 0.0:
            raise ValueError("safety.battery_timeout_s must be positive")

        self.vehicle_status: Optional[Any] = None
        self.local_position: Optional[Any] = None
        self.attitude: Optional[Any] = None
        self.battery_status: Optional[Any] = None
        self.vehicle_status_time = -math.inf
        self.local_position_time = -math.inf
        self.attitude_time = -math.inf
        self.battery_status_time = -math.inf

    def update_vehicle_status(self, message: Any, received_at: float) -> None:
        self.vehicle_status = message
        self.vehicle_status_time = float(received_at)

    def update_local_position(self, message: Any, received_at: float) -> None:
        self.local_position = message
        self.local_position_time = float(received_at)

    def update_attitude(self, message: Any, received_at: float) -> None:
        self.attitude = message
        self.attitude_time = float(received_at)

    def update_battery_status(self, message: Any, received_at: float) -> None:
        self.battery_status = message
        self.battery_status_time = float(received_at)

    def evaluate(self, now: float) -> SafetyResult:
        """Return the current decision and the first failed check."""
        report = self.build_report(now)
        return SafetyResult(bool(report["safe"]), str(report["reason"]))

    def build_report(self, now: float) -> dict[str, Any]:
        """Return every check plus a JSON-safe telemetry snapshot."""
        checks: list[SafetyCheck] = []

        def add(
            name: str,
            passed: bool,
            detail: str,
            failure_reason: str,
        ) -> None:
            checks.append(
                SafetyCheck(name, bool(passed), detail, failure_reason)
            )

        self._add_message_checks(
            checks,
            name="vehicle_status",
            label="vehicle status",
            message=self.vehicle_status,
            received_at=self.vehicle_status_time,
            timeout_s=self.input_timeout_s,
            now=now,
        )
        self._add_message_checks(
            checks,
            name="local_position",
            label="local position",
            message=self.local_position,
            received_at=self.local_position_time,
            timeout_s=self.input_timeout_s,
            now=now,
        )
        self._add_message_checks(
            checks,
            name="vehicle_attitude",
            label="vehicle attitude",
            message=self.attitude,
            received_at=self.attitude_time,
            timeout_s=self.input_timeout_s,
            now=now,
        )
        self._add_battery_message_checks(checks, now)

        status = self.vehicle_status
        failsafe = self._optional_bool(status, "failsafe")
        add(
            "px4_failsafe_clear",
            failsafe is False,
            self._value_detail("failsafe", failsafe),
            "PX4 failsafe is active" if failsafe else "vehicle status has not been received",
        )
        failure_detector = self._optional_int(status, "failure_detector_status")
        add(
            "failure_detector_clear",
            failure_detector == 0,
            self._value_detail("failure_detector_status", failure_detector),
            (
                "PX4 failure detector reports a failure"
                if failure_detector not in (None, 0)
                else "vehicle status has not been received"
            ),
        )

        position = self.local_position
        xy_valid = self._optional_bool(position, "xy_valid")
        z_valid = self._optional_bool(position, "z_valid")
        add(
            "local_xy_valid",
            xy_valid is True,
            self._value_detail("xy_valid", xy_valid),
            "PX4 local position is invalid",
        )
        add(
            "local_z_valid",
            z_valid is True,
            self._value_detail("z_valid", z_valid),
            "PX4 local position is invalid",
        )

        xyz = self._position_xyz(position)
        xyz_finite = xyz is not None and all(math.isfinite(value) for value in xyz)
        add(
            "local_position_finite",
            xyz_finite,
            f"position_ned_m={self._json_vector(xyz)}",
            "PX4 local position contains a non-finite value",
        )

        dead_reckoning = self._optional_bool(position, "dead_reckoning")
        dead_reckoning_ok = dead_reckoning is not None and (
            not dead_reckoning or self.allow_dead_reckoning
        )
        add(
            "dead_reckoning_allowed",
            dead_reckoning_ok,
            (
                f"dead_reckoning={self._format_value(dead_reckoning)}, "
                f"allow_dead_reckoning={str(self.allow_dead_reckoning).lower()}"
            ),
            "PX4 local position is dead reckoning",
        )

        quaternion = self._quaternion(self.attitude)
        quaternion_valid = quaternion is not None and all(
            math.isfinite(value) for value in quaternion
        )
        add(
            "attitude_quaternion_valid",
            quaternion_valid,
            f"q={self._json_vector(quaternion)}",
            "PX4 attitude quaternion is invalid",
        )
        quaternion_norm = (
            math.sqrt(sum(value * value for value in quaternion))
            if quaternion_valid and quaternion is not None
            else math.nan
        )
        add(
            "attitude_quaternion_nonzero",
            math.isfinite(quaternion_norm) and quaternion_norm > 1e-9,
            f"norm={self._format_number(quaternion_norm)}",
            "PX4 attitude quaternion has zero norm",
        )

        self._add_battery_value_checks(checks)

        # 找出所有没有通过的检查
        failed_checks = [
            check
            for check in checks
            if not check.passed
        ]

        # 提取所有错误原因，并删除重复内容
        failure_reasons = list(
            dict.fromkeys(
                check.failure_reason
                for check in failed_checks
                if check.failure_reason
            )
        )

        # 将全部错误原因合并成一个字符串
        result = SafetyResult(
            safe=not failed_checks,
            reason=(
                "all configured safety checks passed"
                if not failed_checks
                else "; ".join(failure_reasons)
            ),
        )

        return {
            "safe": result.safe,
            "reason": result.reason,

            # 字符串列表形式的全部错误
            "errors": failure_reasons,

            # 每一项安全检查的详细结果
            "checks": [
                check.as_dict()
                for check in checks
            ],

            "telemetry": self._telemetry_snapshot(now),
        }

    def _add_message_checks(
        self,
        checks: list[SafetyCheck],
        *,
        name: str,
        label: str,
        message: Any,
        received_at: float,
        timeout_s: float,
        now: float,
    ) -> None:
        received = message is not None
        checks.append(
            SafetyCheck(
                f"{name}_received",
                received,
                "received" if received else "not received",
                f"{label} has not been received",
            )
        )
        age = self._message_age(message, received_at, now)
        fresh = received and age is not None and age <= timeout_s
        checks.append(
            SafetyCheck(
                f"{name}_fresh",
                fresh,
                f"age_s={self._format_number(age)}, limit_s={timeout_s:.3f}",
                f"{label} is stale" if received else f"{label} has not been received",
            )
        )

    def _add_battery_message_checks(
        self, checks: list[SafetyCheck], now: float
    ) -> None:
        if not self.require_battery:
            detail = "disabled by require_battery=false"
            checks.extend(
                (
                    SafetyCheck("battery_status_received", True, detail, ""),
                    SafetyCheck("battery_status_fresh", True, detail, ""),
                )
            )
            return
        self._add_message_checks(
            checks,
            name="battery_status",
            label="battery status",
            message=self.battery_status,
            received_at=self.battery_status_time,
            timeout_s=self.battery_timeout_s,
            now=now,
        )

    def _add_battery_value_checks(self, checks: list[SafetyCheck]) -> None:
        if not self.require_battery:
            detail = "disabled by require_battery=false"
            checks.extend(
                (
                    SafetyCheck("battery_connected", True, detail, ""),
                    SafetyCheck("battery_warning_clear", True, detail, ""),
                    SafetyCheck("battery_remaining_available", True, detail, ""),
                    SafetyCheck("battery_remaining_sufficient", True, detail, ""),
                )
            )
            return

        battery = self.battery_status
        connected = self._optional_bool(battery, "connected")
        checks.append(
            SafetyCheck(
                "battery_connected",
                connected is True,
                self._value_detail("connected", connected),
                "battery is not connected",
            )
        )
        warning = self._optional_int(battery, "warning")
        checks.append(
            SafetyCheck(
                "battery_warning_clear",
                warning is not None and warning < 2,
                self._value_detail("warning", warning),
                "battery warning is critical or worse",
            )
        )
        remaining = self._optional_float(battery, "remaining")
        available = remaining is not None and math.isfinite(remaining) and remaining >= 0.0
        checks.append(
            SafetyCheck(
                "battery_remaining_available",
                available,
                self._value_detail("remaining", self._json_number(remaining)),
                "battery remaining value is unavailable",
            )
        )
        checks.append(
            SafetyCheck(
                "battery_remaining_sufficient",
                available
                and remaining is not None
                and remaining >= self.minimum_battery_remaining,
                (
                    f"remaining={self._format_percent(remaining)}, "
                    f"minimum={self.minimum_battery_remaining * 100.0:.1f}%"
                ),
                "battery remaining is below the configured minimum",
            )
        )

    def _telemetry_snapshot(self, now: float) -> dict[str, Any]:
        status = self.vehicle_status
        position = self.local_position
        attitude = self.attitude
        battery = self.battery_status
        return {
            "message_age_s": {
                "vehicle_status": self._json_number(
                    self._message_age(status, self.vehicle_status_time, now)
                ),
                "local_position": self._json_number(
                    self._message_age(position, self.local_position_time, now)
                ),
                "vehicle_attitude": self._json_number(
                    self._message_age(attitude, self.attitude_time, now)
                ),
                "battery_status": self._json_number(
                    self._message_age(battery, self.battery_status_time, now)
                ),
            },
            "arming_state": self._optional_int(status, "arming_state"),
            "nav_state": self._optional_int(status, "nav_state"),
            "failsafe": self._optional_bool(status, "failsafe"),
            "failure_detector_status": self._optional_int(
                status, "failure_detector_status"
            ),
            "position_ned_m": self._json_vector(self._position_xyz(position)),
            "xy_valid": self._optional_bool(position, "xy_valid"),
            "z_valid": self._optional_bool(position, "z_valid"),
            "dead_reckoning": self._optional_bool(position, "dead_reckoning"),
            "attitude_q": self._json_vector(self._quaternion(attitude)),
            "battery_connected": self._optional_bool(battery, "connected"),
            "battery_warning": self._optional_int(battery, "warning"),
            "battery_remaining": self._json_number(
                self._optional_float(battery, "remaining")
            ),
        }

    @staticmethod
    def _message_age(message: Any, received_at: float, now: float) -> Optional[float]:
        if message is None or not math.isfinite(received_at):
            return None
        return max(0.0, float(now) - received_at)

    @staticmethod
    def _optional_bool(message: Any, field: str) -> Optional[bool]:
        if message is None or not hasattr(message, field):
            return None
        return bool(getattr(message, field))

    @staticmethod
    def _optional_int(message: Any, field: str) -> Optional[int]:
        if message is None or not hasattr(message, field):
            return None
        try:
            return int(getattr(message, field))
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _optional_float(message: Any, field: str) -> Optional[float]:
        if message is None or not hasattr(message, field):
            return None
        try:
            return float(getattr(message, field))
        except (TypeError, ValueError, OverflowError):
            return None

    @classmethod
    def _position_xyz(cls, message: Any) -> Optional[list[float]]:
        if message is None:
            return None
        values = [cls._optional_float(message, field) for field in ("x", "y", "z")]
        if any(value is None for value in values):
            return None
        return [float(value) for value in values if value is not None]

    @classmethod
    def _quaternion(cls, message: Any) -> Optional[list[float]]:
        if message is None:
            return None
        try:
            values = [float(value) for value in getattr(message, "q", [])]
        except (TypeError, ValueError, OverflowError):
            return None
        return values if len(values) == 4 else None

    @staticmethod
    def _json_number(value: Optional[float]) -> Optional[float]:
        if value is None or not math.isfinite(value):
            return None
        return float(value)

    @classmethod
    def _json_vector(cls, values: Optional[list[float]]) -> Optional[list[Optional[float]]]:
        if values is None:
            return None
        return [cls._json_number(value) for value in values]

    @staticmethod
    def _format_value(value: Any) -> str:
        if value is None:
            return "unknown"
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    @classmethod
    def _value_detail(cls, name: str, value: Any) -> str:
        return f"{name}={cls._format_value(value)}"

    @staticmethod
    def _format_number(value: Optional[float]) -> str:
        if value is None or not math.isfinite(value):
            return "unknown"
        return f"{value:.3f}"

    @staticmethod
    def _format_percent(value: Optional[float]) -> str:
        if value is None or not math.isfinite(value) or value < 0.0:
            return "unknown"
        return f"{value * 100.0:.1f}%"
