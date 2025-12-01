# environment.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Point = Tuple[float, float]


def _rect_bounds(xmin: float, ymin: float, xmax: float, ymax: float) -> Dict[str, List[List[float]]]:
    """Axis-aligned rectangle as half-spaces A x <= b."""
    A = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    b = [xmax, -xmin, ymax, -ymin]
    return {"A": A, "b": b}


def _normalize_formation(s: str) -> str:
    """Normalize to the first valid letter among {U, T, A, S, I, N}."""
    if not s:
        raise ValueError("formation must be a non-empty string.")
    for ch in s.upper():
        if ch in "UTASIN":
            return ch
    raise ValueError(f"Unsupported formation: {s!r}. Use one of U,T,A,S,I,N.")

# Minimal letter templates (unit square); each length = minimal readable count.
LETTER_POINTS: Dict[str, List[Point]] = {
    # U (10 pts): symmetric about x=0.5, left column 3, bottom arc 4, right column 3
    "U": [
        (0.20, 0.85), (0.20, 0.60), (0.20, 0.35),                # left column (3)
        (0.30, 0.20), (0.45, 0.15), (0.70, 0.20), (0.55, 0.15),  # bottom arc (4, symmetric about x=0.5)
        (0.80, 0.35), (0.80, 0.60), (0.80, 0.85),                # right column (3, mirror of left)
    ],

    # T (10 pts): top bar 5, stem 5
    "T": [
        (0.15, 0.90), (0.30, 0.90), (0.50, 0.90), (0.70, 0.90), (0.85, 0.90),  # top bar (5)
        (0.50, 0.75), (0.50, 0.60), (0.50, 0.45), (0.50, 0.30), (0.50, 0.15),  # stem (5)
    ],

    # A (10 pts): top 1, upper-mid 2, mid 3, lower-mid 2, bottom 2
    # Symmetric legs with a clear mid bar; y-levels chosen for visual balance.
    "A": [
        (0.50, 0.90),                          # top (apex) - 1
        (0.44, 0.72), (0.56, 0.72),            # upper-mid along legs - 2
        (0.42, 0.55), (0.50, 0.55), (0.58, 0.55),  # mid bar (L-center-R) - 3
        (0.36, 0.38), (0.64, 0.38),            # lower-mid along legs - 2
        (0.28, 0.12), (0.72, 0.12),            # bottom corners - 2
    ],

    # S (10 pts): symmetric about center (0.5, 0.5), top half 5, bottom half 5
    # For point (x, y), mirror is (1-x, 1-y) about center (0.5, 0.5)
    "S": [
        (0.70, 0.85),  # 1 top-right bulge -> mirrors to (0.30, 0.15)
        (0.50, 0.95),  # 2 top-center -> mirrors to (0.50, 0.05)
        (0.30, 0.85),  # 3 top-left -> mirrors to (0.70, 0.15)
        (0.30, 0.70),  # 4 upper-left curve -> mirrors to (0.70, 0.30)
        (0.43, 0.60),  # 5 upper-center -> mirrors to (0.50, 0.40)
        (0.57, 0.40),  # 6 lower-center (mirror of 5: 1-0.5=0.5, 1-0.6=0.4)
        (0.70, 0.30),  # 7 lower-right curve (mirror of 4: 1-0.3=0.7, 1-0.7=0.3)
        (0.70, 0.15),  # 8 bottom-right (mirror of 3: 1-0.3=0.7, 1-0.85=0.15)
        (0.50, 0.05),  # 9 bottom-center (mirror of 2: 1-0.5=0.5, 1-0.95=0.05)
        (0.30, 0.15),  # 10 bottom-left bulge (mirror of 1: 1-0.7=0.3, 1-0.85=0.15)
    ],

    # I (10 pts): top bar 3, vertical 4, bottom bar 3
    "I": [
        (0.35, 0.90), (0.50, 0.90), (0.65, 0.90),  # top bar (3)
        (0.50, 0.75), (0.50, 0.60), (0.50, 0.45), (0.50, 0.30),  # vertical (4)
        (0.35, 0.10), (0.50, 0.10), (0.65, 0.10),  # bottom bar (3)
    ],

    # N (10 pts): left column 4, diagonal 3, right column 3
    "N": [
        # left column (4) -- x = 0.25
        (0.25, 0.12), (0.25, 0.38), (0.25, 0.64), (0.25, 0.88),
        # diagonal (3) -- from left-bottom toward right-top
        (0.42, 0.34), (0.50, 0.50), (0.58, 0.66),
        # right column (3) -- x = 0.75 (mirror of left)
        (0.75, 0.12), (0.75, 0.50), (0.75, 0.88),
    ],
}
MIN_COUNT: Dict[str, int] = {k: len(v) for k, v in LETTER_POINTS.items()}


def _extract_rect(bounds: Dict[str, List[List[float]]]) -> Tuple[float, float, float, float]:
    """Extract (xmin, ymin, xmax, ymax) from our rectangle A,b format; fallback to default."""
    A = bounds.get("A", [])
    b = bounds.get("b", [])
    if len(A) == 4 and len(b) == 4:
        xmax = float(b[0]); xmin = -float(b[1])
        ymax = float(b[2]); ymin = -float(b[3])
        return xmin, ymin, xmax, ymax
    return -5.0, -3.0, 5.0, 3.0


def _fit_points_to_bounds(points: List[Point], bounds: Dict[str, List[List[float]]]) -> List[Point]:
    """Affine-map unit-square points into the rectangle described by bounds (A x <= b)."""
    xmin, ymin, xmax, ymax = _extract_rect(bounds)
    mx, my = 0.12, 0.12  # margin fraction
    W = (1.0 - 2 * mx) * (xmax - xmin)
    H = (1.0 - 2 * my) * (ymax - ymin)
    tx = xmin + mx * (xmax - xmin)
    ty = ymin + my * (ymax - ymin)
    aspect = 1.05  # slight vertical boost
    H *= aspect
    ty = (ymin + ymax - H) * 0.5
    return [(tx + x * W, ty + y * H) for (x, y) in points]


def _corner_fill_points(bounds: Dict[str, List[List[float]]], k: int) -> List[Point]:
    """
    Generate k extra points stacked near the bottom-right corner of the bounds.
    Layout: up to 5 points per column (bottom to top), then open a new column to the left.
    """
    if k <= 0:
        return []
    xmin, ymin, xmax, ymax = _extract_rect(bounds)
    # Local placement box inside the rectangle
    mx, my = 0.08, 0.08               # corner margin fraction
    gx, gy = 0.08, 0.09               # spacing fraction (x leftward, y upward)
    W = xmax - xmin
    H = ymax - ymin
    # Anchor at bottom-right corner inside margins
    x0 = xmax - mx * W
    y0 = ymin + my * H
    pts: List[Point] = []
    col_height = 5                     # max items per column
    for i in range(k):
        col = i // col_height
        row = i % col_height
        x = x0 - col * gx * W
        y = y0 + row * gy * H
        # Keep strictly inside the rectangle
        x = min(max(x, xmin + 1e-6), xmax - 1e-6)
        y = min(max(y, ymin + 1e-6), ymax - 1e-6)
        pts.append((x, y))
    return pts

def generate_circle_goals(
    num_robot: int,
    robot_radius: float,
    bounds: Dict[str, List[List[float]]],
    margin_frac: float = 0.12,
    angle_offset: float = 0.0) -> List[Point]:
    """
    Generate a single-ring (circle) of goals uniformly spaced inside `bounds`.

    Rules:
      1) Neighbor spacing: chord length >= 2 * robot_radius
         => 2*R*sin(pi/n) >= 2*robot_radius => R >= robot_radius / sin(pi/n).
      2) Wall clearance: the ring plus robot footprint must fit within `bounds`,
         leaving a fractional margin `margin_frac` (same convention as _fit_points_to_bounds).

    Returns:
      List of (x, y) positions with length == num_robot.
    """
    import math

    if num_robot <= 0:
        raise ValueError(f"num_robot must be positive; got {num_robot}.")
    if robot_radius <= 0:
        raise ValueError(f"robot_radius must be positive; got {robot_radius}.")

    xmin, ymin, xmax, ymax = _extract_rect(bounds)
    W = xmax - xmin
    H = ymax - ymin
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    # Max allowable ring radius from walls (keep margin and robot fully inside).
    R_max_x = (0.5 - margin_frac) * W - robot_radius
    R_max_y = (0.5 - margin_frac) * H - robot_radius
    R_max = max(0.0, min(R_max_x, R_max_y))

    # If only one robot, place it at the center.
    if num_robot == 1:
        return [(cx, cy)]

    # Neighbor spacing constraint.
    sin_term = math.sin(math.pi / num_robot)
    R_required = float("inf") if sin_term <= 1e-9 else (robot_radius / sin_term)

    # A small safety floor to avoid degenerate radii.
    R_floor = 10 * robot_radius
    R = max(R_required, R_floor)

    if R > R_max + 1e-9:
        raise ValueError(
            f"Cannot place {num_robot} robots of radius {robot_radius:g} on a ring inside bounds: "
            f"required R >= {R_required:.3f}, but max allowed is {R_max:.3f}."
        )

    # Evenly distribute points on the circle.
    pts: List[Point] = []
    for k in range(num_robot):
        theta = angle_offset + 2.0 * math.pi * (k / num_robot)
        x = cx + R * math.cos(theta)
        y = cy + R * math.sin(theta)
        # Final safety clamp within bounds.
        x = min(max(x, xmin + 1e-6), xmax - 1e-6)
        y = min(max(y, ymin + 1e-6), ymax - 1e-6)
        pts.append((x, y))
    return pts


@dataclass
class Environment:
    """
    Minimal environment for letter formations (U, T, A, S, I, N).

    Members:
      - formation: str, normalized to one of {'U','T','A','S','I','N'}.
      - num_robot: int, total robots to place (can exceed minimal; extras go to bottom-right).
      - goals: list of (x, y) positions.
      - bounds: dict with 'A' and 'b' for polygon Ax <= b (rectangle by default).

    Rules:
      - Minimal robots per letter given by MIN_COUNT.
      - max_num_robot defaults to 12.
      - If num_robot < minimal -> ValueError.
      - If num_robot > minimal -> extra goals are placed at the bottom-right corner.
    """
    formation: str
    num_robot: int | None = None
    goals: List[Point] = field(default_factory=list)
    bounds: Dict[str, List[List[float]]] = field(default_factory=dict)
    max_num_robot: int = 12

    def __post_init__(self):
        self.formation = _normalize_formation(self.formation)
        if not self.bounds:
            self.bounds = _rect_bounds(-5.0, -3.0, 5.0, 3.0)
        minimal = MIN_COUNT[self.formation]
        template = LETTER_POINTS[self.formation]

        # Decide total robots
        if self.num_robot is None:
            self.num_robot = minimal
        if self.num_robot < minimal:
            raise ValueError(f"{self.formation} needs at least {minimal} robots; got {self.num_robot}.")
        if self.num_robot > self.max_num_robot:
            self.num_robot = self.max_num_robot  # clamp global maximum

        # Base letter goals (minimal set)
        base = _fit_points_to_bounds(template, self.bounds)

        # Extra robots -> bottom-right corner padding
        extra_n = max(0, self.num_robot - minimal)
        extra = _corner_fill_points(self.bounds, extra_n)

        self.goals = base + extra

    def set_rect_bounds(self, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        """Set rectangular bounds and regenerate goals (keep extras at bottom-right)."""
        self.bounds = _rect_bounds(xmin, ymin, xmax, ymax)
        minimal = MIN_COUNT[self.formation]
        template = LETTER_POINTS[self.formation]
        base = _fit_points_to_bounds(template, self.bounds)
        extra_n = max(0, self.num_robot - minimal)
        extra = _corner_fill_points(self.bounds, extra_n)
        self.goals = base + extra

    def change_formation(self, formation: str, num_robot: int | None = None) -> None:
        """Change letter and (optionally) total robots, then regenerate goals."""
        self.formation = _normalize_formation(formation)
        if num_robot is not None:
            self.num_robot = num_robot
        minimal = MIN_COUNT[self.formation]
        if self.num_robot < minimal:
            raise ValueError(f"{self.formation} needs at least {minimal} robots; got {self.num_robot}.")
        if self.num_robot > self.max_num_robot:
            self.num_robot = self.max_num_robot
        template = LETTER_POINTS[self.formation]
        base = _fit_points_to_bounds(template, self.bounds)
        extra_n = max(0, self.num_robot - minimal)
        extra = _corner_fill_points(self.bounds, extra_n)
        self.goals = base + extra