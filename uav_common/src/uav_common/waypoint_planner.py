"""Pure-Python lawnmower waypoint generation for a rectangular NED area."""

from __future__ import annotations

import math
from typing import Sequence


Point2D = tuple[float, float]


class WaypointPlanner:
    """Generate absolute local-NED XY scan-line endpoints for a rectangle.

    The four input corners may be supplied in any order.  The longer side of
    the rectangle is used as the sweep direction and the shorter side is used
    as the lateral direction.  Safety margin and lateral step are fixed class
    settings; callers only provide the four rectangle corners.
    """

    safety_margin_m = 0.3
    lateral_step_m = 0.5

    # This tolerance is only for floating-point geometry comparisons.
    _geometry_tolerance = 1e-6

    def plan(
        self,
        corners_ned_xy: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """Return paired ``[north, east]`` endpoints without a flight order."""
        corners = self._parse_corners(corners_ned_xy)
        ordered = self._order_corners(corners)
        edges = [
            self._subtract(ordered[(index + 1) % 4], ordered[index])
            for index in range(4)
        ]
        lengths = [self._norm(edge) for edge in edges]
        self._validate_rectangle(edges, lengths)

        # Adjacent edges describe the two rectangle axes.  Sweep along the
        # longer one to minimize the number of turns.
        sweep_edge = edges[0] if lengths[0] >= lengths[1] else edges[1]
        sweep_length = self._norm(sweep_edge)
        sweep_axis = (
            sweep_edge[0] / sweep_length,
            sweep_edge[1] / sweep_length,
        )
        sweep_axis = self._canonical_direction(sweep_axis)
        lateral_axis = (-sweep_axis[1], sweep_axis[0])

        centre = (
            sum(point[0] for point in ordered) / 4.0,
            sum(point[1] for point in ordered) / 4.0,
        )
        sweep_projections = [
            self._dot(self._subtract(point, centre), sweep_axis)
            for point in ordered
        ]
        lateral_projections = [
            self._dot(self._subtract(point, centre), lateral_axis)
            for point in ordered
        ]

        sweep_min = min(sweep_projections) + self.safety_margin_m
        sweep_max = max(sweep_projections) - self.safety_margin_m
        lateral_min = min(lateral_projections) + self.safety_margin_m
        lateral_max = max(lateral_projections) - self.safety_margin_m

        lateral_positions = self._lane_positions(lateral_min, lateral_max)
        waypoints: list[list[float]] = []
        for lateral_position in lateral_positions:
            first = self._from_local_axes(
                centre,
                sweep_axis,
                lateral_axis,
                sweep_min,
                lateral_position,
            )
            second = self._from_local_axes(
                centre,
                sweep_axis,
                lateral_axis,
                sweep_max,
                lateral_position,
            )
            # Every adjacent pair belongs to one scan line.  All pairs use the
            # same geometric direction; no flight start or direction is
            # selected here.
            waypoints.extend(([first[0], first[1]], [second[0], second[1]]))

        return waypoints

    def _lane_positions(self, start: float, end: float) -> list[float]:
        positions = [start]
        candidate = start + self.lateral_step_m
        while candidate < end - self._geometry_tolerance:
            positions.append(candidate)
            candidate += self.lateral_step_m

        if not math.isclose(
            positions[-1],
            end,
            rel_tol=self._geometry_tolerance,
            abs_tol=self._geometry_tolerance,
        ):
            positions.append(end)
        return positions

    def _parse_corners(
        self,
        corners_ned_xy: Sequence[Sequence[float]],
    ) -> list[Point2D]:
        if len(corners_ned_xy) != 4:
            raise ValueError("corners_ned_xy must contain exactly four points")

        corners: list[Point2D] = []
        for index, point in enumerate(corners_ned_xy):
            if len(point) != 2:
                raise ValueError(
                    f"corners_ned_xy[{index}] must contain [north, east]"
                )
            north = self._finite_number(point[0], f"corners_ned_xy[{index}][0]")
            east = self._finite_number(point[1], f"corners_ned_xy[{index}][1]")
            corners.append((north, east))

        if len(set(corners)) != 4:
            raise ValueError("corners_ned_xy must contain four unique points")
        return corners

    def _order_corners(self, corners: Sequence[Point2D]) -> list[Point2D]:
        centre = (
            sum(point[0] for point in corners) / 4.0,
            sum(point[1] for point in corners) / 4.0,
        )
        ordered = sorted(
            corners,
            key=lambda point: math.atan2(
                point[1] - centre[1], point[0] - centre[0]
            ),
        )

        twice_area = sum(
            ordered[index][0] * ordered[(index + 1) % 4][1]
            - ordered[(index + 1) % 4][0] * ordered[index][1]
            for index in range(4)
        )
        if abs(twice_area) <= self._geometry_tolerance:
            raise ValueError("corners_ned_xy do not form a non-degenerate rectangle")

        # Rotate the cycle to make output independent of the input order.
        first_index = min(range(4), key=lambda index: ordered[index])
        return ordered[first_index:] + ordered[:first_index]

    def _validate_rectangle(
        self,
        edges: Sequence[Point2D],
        lengths: Sequence[float],
    ) -> None:
        if any(length <= self._geometry_tolerance for length in lengths):
            raise ValueError("rectangle edges must have non-zero length")

        tolerance = self._geometry_tolerance
        for index in range(4):
            current = edges[index]
            following = edges[(index + 1) % 4]
            scale = lengths[index] * lengths[(index + 1) % 4]
            if abs(self._dot(current, following)) > tolerance * scale:
                raise ValueError("corners_ned_xy must form a rectangle")

        if not math.isclose(
            lengths[0], lengths[2], rel_tol=tolerance, abs_tol=tolerance
        ) or not math.isclose(
            lengths[1], lengths[3], rel_tol=tolerance, abs_tol=tolerance
        ):
            raise ValueError("opposite rectangle edges must have equal length")

    @staticmethod
    def _canonical_direction(axis: Point2D) -> Point2D:
        if axis[0] < 0.0 or (math.isclose(axis[0], 0.0) and axis[1] < 0.0):
            return (-axis[0], -axis[1])
        return axis

    @staticmethod
    def _from_local_axes(
        centre: Point2D,
        sweep_axis: Point2D,
        lateral_axis: Point2D,
        sweep_position: float,
        lateral_position: float,
    ) -> Point2D:
        return (
            centre[0]
            + sweep_position * sweep_axis[0]
            + lateral_position * lateral_axis[0],
            centre[1]
            + sweep_position * sweep_axis[1]
            + lateral_position * lateral_axis[1],
        )

    @staticmethod
    def _finite_number(value: float, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"{name} must be a finite number")
        return parsed

    @staticmethod
    def _subtract(first: Point2D, second: Point2D) -> Point2D:
        return first[0] - second[0], first[1] - second[1]

    @staticmethod
    def _dot(first: Point2D, second: Point2D) -> float:
        return first[0] * second[0] + first[1] * second[1]

    @staticmethod
    def _norm(vector: Point2D) -> float:
        return math.hypot(vector[0], vector[1])


class WaypointOrderPlanner:
    """Build a serpentine route starting at the nearest of four corners."""

    def order(
        self,
        vehicle_ned: Sequence[float],
        waypoints_ned_xy: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """Return a copied route ordered from the corner nearest the vehicle."""
        if len(vehicle_ned) != 3:
            raise ValueError("vehicle_ned must contain [north, east, down]")
        if any(isinstance(value, bool) for value in vehicle_ned):
            raise ValueError("vehicle_ned must contain finite numbers")
        try:
            vehicle = [float(value) for value in vehicle_ned]
        except (TypeError, ValueError) as exc:
            raise ValueError("vehicle_ned must contain finite numbers") from exc
        if not all(math.isfinite(value) for value in vehicle):
            raise ValueError("vehicle_ned must contain finite numbers")

        waypoints: list[list[float]] = []
        for index, point in enumerate(waypoints_ned_xy):
            if len(point) != 2 or any(isinstance(value, bool) for value in point):
                raise ValueError(
                    f"waypoints_ned_xy[{index}] must contain [north, east]"
                )
            try:
                parsed = [float(point[0]), float(point[1])]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"waypoints_ned_xy[{index}] must contain finite numbers"
                ) from exc
            if not all(math.isfinite(value) for value in parsed):
                raise ValueError(
                    f"waypoints_ned_xy[{index}] must contain finite numbers"
                )
            waypoints.append(parsed)

        if not waypoints:
            return waypoints
        if len(waypoints) % 2 != 0:
            raise ValueError(
                "waypoints_ned_xy must contain pairs of scan-line endpoints"
            )

        lanes = [waypoints[index : index + 2] for index in range(0, len(waypoints), 2)]

        def build_route(
            ordered_lanes: Sequence[Sequence[list[float]]],
            start_at_second_endpoint: bool,
        ) -> list[list[float]]:
            route: list[list[float]] = []
            initial_start_index = 1 if start_at_second_endpoint else 0
            for lane_index, lane in enumerate(ordered_lanes):
                start_index = (
                    initial_start_index
                    if lane_index % 2 == 0
                    else 1 - initial_start_index
                )
                route.extend((lane[start_index], lane[1 - start_index]))
            return route

        forward_lanes = lanes
        reverse_lanes = list(reversed(lanes))
        candidates = [
            build_route(forward_lanes, False),
            build_route(forward_lanes, True),
            build_route(reverse_lanes, False),
            build_route(reverse_lanes, True),
        ]

        def start_distance_squared(route: Sequence[Sequence[float]]) -> float:
            return (
                (route[0][0] - vehicle[0]) ** 2
                + (route[0][1] - vehicle[1]) ** 2
            )

        return min(candidates, key=start_distance_squared)


class LatLonToNedConverter:
    """Convert four latitude/longitude corners to absolute local-NED XY."""

    _wgs84_semi_major_axis_m = 6_378_137.0
    _wgs84_flattening = 1.0 / 298.257223563

    def convert(
        self,
        corners_lat_lon: Sequence[Sequence[float]],
        home_lat_lon: Sequence[float],
        home_ned: Sequence[float],
    ) -> list[list[float]]:
        """Return four ``[north, east]`` points in the UAV local-NED frame."""
        if len(corners_lat_lon) != 4:
            raise ValueError("corners_lat_lon must contain exactly four points")

        home_latitude, home_longitude = self._parse_lat_lon(
            home_lat_lon, "home_lat_lon"
        )
        if len(home_ned) != 3 or any(isinstance(value, bool) for value in home_ned):
            raise ValueError("home_ned must contain [north, east, down]")
        try:
            parsed_home_ned = [float(value) for value in home_ned]
        except (TypeError, ValueError) as exc:
            raise ValueError("home_ned must contain finite numbers") from exc
        if not all(math.isfinite(value) for value in parsed_home_ned):
            raise ValueError("home_ned must contain finite numbers")

        latitude_rad = math.radians(home_latitude)
        sin_latitude = math.sin(latitude_rad)
        eccentricity_squared = self._wgs84_flattening * (
            2.0 - self._wgs84_flattening
        )
        denominator = math.sqrt(
            1.0 - eccentricity_squared * sin_latitude * sin_latitude
        )
        meridian_radius_m = (
            self._wgs84_semi_major_axis_m
            * (1.0 - eccentricity_squared)
            / (denominator**3)
        )
        prime_vertical_radius_m = self._wgs84_semi_major_axis_m / denominator

        corners_ned_xy: list[list[float]] = []
        for index, point in enumerate(corners_lat_lon):
            latitude, longitude = self._parse_lat_lon(
                point, f"corners_lat_lon[{index}]"
            )
            latitude_delta_rad = math.radians(latitude - home_latitude)
            longitude_delta_deg = (
                (longitude - home_longitude + 180.0) % 360.0
            ) - 180.0
            longitude_delta_rad = math.radians(longitude_delta_deg)

            north_offset_m = latitude_delta_rad * meridian_radius_m
            east_offset_m = (
                longitude_delta_rad
                * prime_vertical_radius_m
                * math.cos(latitude_rad)
            )
            corners_ned_xy.append(
                [
                    parsed_home_ned[0] + north_offset_m,
                    parsed_home_ned[1] + east_offset_m,
                ]
            )

        return corners_ned_xy

    @staticmethod
    def _parse_lat_lon(
        value: Sequence[float],
        name: str,
    ) -> tuple[float, float]:
        if len(value) != 2 or any(isinstance(item, bool) for item in value):
            raise ValueError(f"{name} must contain [latitude, longitude]")
        try:
            latitude = float(value[0])
            longitude = float(value[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain finite numbers") from exc
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            raise ValueError(f"{name} must contain finite numbers")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError(f"{name} latitude must be in [-90, 90]")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError(f"{name} longitude must be in [-180, 180]")
        return latitude, longitude

