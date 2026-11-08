"""Coverage waypoints inside an explicitly closed polygon in local NED."""

from __future__ import annotations

import heapq
import math
from typing import Sequence

from shapely.geometry import LineString, Point, Polygon
from shapely.validation import explain_validity


Point2D = tuple[float, float]


class WaypointPlanner:
    """Generate paired scan endpoints for a simple polygon, including concavities.

    Input contains n + 1 ordered points for n >= 3 edges: the last point must
    repeat the first.  Clockwise and counterclockwise boundaries are accepted;
    vertices are never angle-sorted or automatically closed.  Scan segments
    follow the longest boundary edge and are clipped to the inset polygon.
    """

    safety_margin_m = 0.3
    lateral_step_m = 0.5

    _coordinate_tolerance_m = 1e-6
    _area_tolerance_m2 = 1e-6
    _numeric_tolerance_m = 1e-9

    def plan(
        self,
        corners_ned_xy: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """Return paired ``[north, east]`` endpoints without a flight order."""
        vertices = self.validate_polygon(corners_ned_xy)
        edges = [
            self._subtract(vertices[(index + 1) % len(vertices)], point)
            for index, point in enumerate(vertices)
        ]

        sweep_edge = max(edges, key=self._norm)
        sweep_length = self._norm(sweep_edge)
        sweep_axis = (
            sweep_edge[0] / sweep_length,
            sweep_edge[1] / sweep_length,
        )
        sweep_axis = self._canonical_direction(sweep_axis)
        lateral_axis = (-sweep_axis[1], sweep_axis[0])

        origin = vertices[0]
        local_polygon = Polygon([
            (self._dot(self._subtract(point, origin), sweep_axis),
             self._dot(self._subtract(point, origin), lateral_axis))
            for point in vertices
        ])
        scan_area = self._inset_polygon(local_polygon)
        sweep_min, lateral_min, sweep_max, lateral_max = scan_area.bounds

        lateral_positions = self._lane_positions(lateral_min, lateral_max)
        # A projected horizontal edge can acquire a tiny slope.  Move extreme
        # scan lines just inside so a boundary lane does not become a point.
        roundoff = min(
            max(self._numeric_tolerance_m,
                8 * math.ulp(max(abs(lateral_min), abs(lateral_max)))),
            (lateral_max - lateral_min) / 4.0,
        )
        waypoints: list[list[float]] = []
        for lateral_position in lateral_positions:
            lateral_position = min(
                max(lateral_position, lateral_min + roundoff),
                lateral_max - roundoff,
            )
            scan_line = LineString([
                (sweep_min, lateral_position), (sweep_max, lateral_position)
            ])
            segments = self._line_segments(scan_area.intersection(scan_line))
            intervals = sorted(
                (min(point[0] for point in segment.coords),
                 max(point[0] for point in segment.coords))
                for segment in segments
                if segment.length > self._coordinate_tolerance_m
            )
            # A concave boundary may split one scan line into several spans.
            for start, end in intervals:
                for sweep_position in (start, end):
                    point = self._from_local_axes(
                        origin, sweep_axis, lateral_axis,
                        sweep_position, lateral_position,
                    )
                    waypoints.append([point[0], point[1]])

        if not waypoints:
            raise ValueError("polygon has no usable scan lines after safety margin")
        return waypoints

    def validate_polygon(
        self, corners_ned_xy: Sequence[Sequence[float]],
    ) -> list[Point2D]:
        """Validate a closed ring and return its n vertices in input order.

        This only checks the boundary; feasibility after applying the safety
        margin is a separate planning check.
        """
        vertices = self._parse_corners(corners_ned_xy)
        for index, point in enumerate(vertices):
            following = vertices[(index + 1) % len(vertices)]
            if math.dist(point, following) <= self._coordinate_tolerance_m:
                raise ValueError("polygon edges must have non-zero length")
            if any(math.dist(point, previous) <= self._coordinate_tolerance_m
                   for previous in vertices[:index]):
                raise ValueError("polygon vertices must be distinct")

        polygon = Polygon(vertices)
        if not polygon.is_valid:
            raise ValueError(
                "corners_ned_xy must form a simple closed polygon: "
                + explain_validity(polygon)
            )
        if (not math.isfinite(polygon.area)
                or polygon.area <= self._area_tolerance_m2):
            raise ValueError("polygon must have finite non-zero area")
        return vertices

    @staticmethod
    def _line_segments(geometry) -> list[LineString]:
        if geometry.is_empty:
            return []
        if geometry.geom_type == "LineString":
            return [geometry]
        if hasattr(geometry, "geoms"):
            return [line for part in geometry.geoms
                    for line in WaypointPlanner._line_segments(part)]
        return []  # A tangent point does not define a scan segment.

    def _inset_polygon(self, polygon: Polygon) -> Polygon:
        # Mitre joins keep straight edges and conservatively clear concavities.
        inset = polygon.buffer(-self.safety_margin_m, join_style=2)
        if inset.is_empty or inset.area <= self._area_tolerance_m2:
            raise ValueError("polygon has no usable area after safety margin")
        if inset.geom_type != "Polygon" or not inset.is_valid:
            raise ValueError("polygon safety margin leaves disconnected scan areas")
        return inset

    def _lane_positions(self, start: float, end: float) -> list[float]:
        count = int(math.floor((end - start) / self.lateral_step_m))
        positions = [start + index * self.lateral_step_m
                     for index in range(count + 1)]
        if end - positions[-1] > self._numeric_tolerance_m:
            positions.append(end)
        return positions

    def _parse_corners(
        self,
        corners_ned_xy: Sequence[Sequence[float]],
    ) -> list[Point2D]:
        try:
            if len(corners_ned_xy) < 4:
                raise ValueError(
                    "corners_ned_xy must contain at least three vertices "
                    "plus the repeated first point"
                )
        except TypeError as exc:
            raise ValueError("corners_ned_xy must be a sequence of points") from exc

        corners: list[Point2D] = []
        for index, point in enumerate(corners_ned_xy):
            try:
                if isinstance(point, (str, bytes)) or len(point) != 2:
                    raise ValueError()
                north = self._finite_number(point[0], f"corners_ned_xy[{index}][0]")
                east = self._finite_number(point[1], f"corners_ned_xy[{index}][1]")
            except (TypeError, KeyError, IndexError, ValueError) as exc:
                raise ValueError(
                    f"corners_ned_xy[{index}] must contain finite [north, east]"
                ) from exc
            corners.append((north, east))

        if math.dist(corners[0], corners[-1]) > self._coordinate_tolerance_m:
            raise ValueError(
                "corners_ned_xy must be explicitly closed "
                "(last point repeats first)"
            )
        return corners[:-1]

    @staticmethod
    def _canonical_direction(axis: Point2D) -> Point2D:
        if axis[0] < 0.0 or (math.isclose(axis[0], 0.0) and axis[1] < 0.0):
            return (-axis[0], -axis[1])
        return axis

    @staticmethod
    def _from_local_axes(
        origin: Point2D,
        sweep_axis: Point2D,
        lateral_axis: Point2D,
        sweep_position: float,
        lateral_position: float,
    ) -> Point2D:
        return (
            origin[0]
            + sweep_position * sweep_axis[0]
            + lateral_position * lateral_axis[0],
            origin[1]
            + sweep_position * sweep_axis[1]
            + lateral_position * lateral_axis[1],
        )

    @staticmethod
    def _finite_number(value: float, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
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
    """Order scan segments and connect them inside the supplied polygon."""

    def order(
        self,
        vehicle_ned: Sequence[float],
        waypoints_ned_xy: Sequence[Sequence[float]],
        corners_ned_xy: Sequence[Sequence[float]] | None = None,
    ) -> list[list[float]]:
        """Start at the nearest of four traversal entries.

        Supplying a closed polygon inserts internal transit vertices wherever
        a straight connection would leave the inset boundary.  The approach
        from the vehicle to the first waypoint is outside this scan route.
        Omitting the boundary preserves the legacy straight-connection API.
        """
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

        route = min(candidates, key=start_distance_squared)
        if corners_ned_xy is None:
            return route

        planner = WaypointPlanner()
        vertices = planner.validate_polygon(corners_ned_xy)
        origin = vertices[0]
        # Compute offsets close to the origin to reduce translation roundoff.
        area = planner._inset_polygon(Polygon([
            planner._subtract(point, origin) for point in vertices
        ]))
        local_route = [planner._subtract(point, origin) for point in route]
        connected = _PolygonRouter(
            area, planner._coordinate_tolerance_m,
        ).connect(local_route)
        return [[point[0] + origin[0], point[1] + origin[1]]
                for point in connected]


class _PolygonRouter:
    """Use a visibility graph to connect endpoints within the inset boundary."""

    def __init__(self, area: Polygon, tolerance: float) -> None:
        # This allowance is only for projection/offset floating-point error.
        self.area = area.buffer(tolerance, join_style=2)
        self.vertices = list(area.exterior.coords)[:-1]
        for ring in area.interiors:
            self.vertices.extend(list(ring.coords)[:-1])
        self.graph: list[list[tuple[int, float]]] | None = None

    def _visible(self, first: Point2D, second: Point2D) -> bool:
        if first == second:
            return True
        return self.area.covers(LineString([first, second]))

    def _boundary_graph(self) -> list[list[tuple[int, float]]]:
        if self.graph is None:
            self.graph = [[] for _ in self.vertices]
            for index, first in enumerate(self.vertices):
                for following in range(index + 1, len(self.vertices)):
                    second = self.vertices[following]
                    if self._visible(first, second):
                        distance = math.dist(first, second)
                        self.graph[index].append((following, distance))
                        self.graph[following].append((index, distance))
        return self.graph

    def _path(self, start: Point2D, end: Point2D) -> list[Point2D]:
        if self._visible(start, end):
            return [start, end]
        graph = [list(neighbors) for neighbors in self._boundary_graph()]
        nodes = self.vertices + [start, end]
        start_index, end_index = len(self.vertices), len(self.vertices) + 1
        graph.extend([[], []])
        for endpoint in (start_index, end_index):
            for index, vertex in enumerate(self.vertices):
                if self._visible(nodes[endpoint], vertex):
                    distance = math.dist(nodes[endpoint], vertex)
                    graph[endpoint].append((index, distance))
                    graph[index].append((endpoint, distance))

        distances = [math.inf] * len(nodes)
        distances[start_index] = 0.0
        previous: dict[int, int] = {}
        queue = [(0.0, start_index)]
        while queue:
            distance, current = heapq.heappop(queue)
            if distance > distances[current]:
                continue
            if current == end_index:
                path = [end]
                while current != start_index:
                    current = previous[current]
                    path.append(nodes[current])
                return list(reversed(path))
            for following, length in graph[current]:
                candidate = distance + length
                if candidate < distances[following]:
                    distances[following] = candidate
                    previous[following] = current
                    heapq.heappush(queue, (candidate, following))
        raise ValueError("cannot connect scan segments inside polygon safety margin")

    def connect(self, route: Sequence[Sequence[float]]) -> list[list[float]]:
        points = [(point[0], point[1]) for point in route]
        if any(not self.area.covers(Point(point)) for point in points):
            raise ValueError("scan endpoints must lie inside polygon safety margin")
        result = [list(points[0])]
        for first, second in zip(points, points[1:]):
            result.extend(list(point) for point in self._path(first, second)[1:])
        return result


class LatLonToNedConverter:
    """Convert a closed latitude/longitude boundary to absolute local-NED XY."""

    _wgs84_semi_major_axis_m = 6_378_137.0
    _wgs84_flattening = 1.0 / 298.257223563

    def convert(
        self,
        corners_lat_lon: Sequence[Sequence[float]],
        home_lat_lon: Sequence[float],
        home_ned: Sequence[float],
    ) -> list[list[float]]:
        """Return n + 1 points, preserving boundary order and the closing point."""
        if len(corners_lat_lon) < 4:
            raise ValueError(
                "corners_lat_lon needs at least three vertices plus a closing point"
            )

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

        WaypointPlanner().validate_polygon(corners_ned_xy)
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

