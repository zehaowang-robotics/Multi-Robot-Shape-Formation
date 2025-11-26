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
    # U: verticals + rounded bottom; include a central bottom node.
    # Pairwise symmetric points plus the center.
    "U": [
        (0.20, 0.85), (0.20, 0.60), (0.20, 0.35),      # left column
        (0.35, 0.15), (0.50, 0.10), (0.65, 0.15),      # bottom arc (center included)
        (0.80, 0.35), (0.80, 0.60), (0.80, 0.85),      # right column
    ],

    # T: top bar with symmetric samples and a vertical stem.
    # The intersection (0.5, 0.9) is explicitly included.
    "T": [
        (0.15, 0.90), (0.30, 0.90), (0.50, 0.90), (0.70, 0.90), (0.85, 0.90),  # top bar (center included)
        (0.50, 0.70), (0.50, 0.50), (0.50, 0.30), (0.50, 0.15),                # stem
    ],

    # A: two legs + mid bar; apex centered at (0.5, 0.90).
    # Mid bar contains left joint, center, right joint to ensure clear structure.
    "A": [
        (0.25, 0.10), (0.375, 0.50), (0.50, 0.90), (0.625, 0.50), (0.75, 0.10),  # legs (symmetric + apex)
        (0.40, 0.55), (0.50, 0.55), (0.60, 0.55),                                 # mid bar (L-joint, center, R-joint)
        (0.50, 0.70),                                                             # inner support along the spine
    ],

    # S / I / N kept as before (no symmetry constraint requested).
    "S": [
        (0.80, 0.85), (0.60, 0.95), (0.35, 0.90), (0.20, 0.75), (0.40, 0.60),
        (0.60, 0.50), (0.80, 0.35), (0.60, 0.20), (0.30, 0.15),
    ],
    "I": [
        (0.50, 0.90), (0.50, 0.50), (0.50, 0.10),
    ],
    "N": [
        (0.20, 0.10), (0.20, 0.40), (0.20, 0.70),
        (0.35, 0.75), (0.55, 0.45), (0.75, 0.20),
        (0.80, 0.50), (0.80, 0.85),
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
