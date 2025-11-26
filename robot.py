# robot.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Callable
import math


def _normalize_steering_type(s: str) -> str:
    """
    Normalize user-provided steering_type into canonical tokens.
    Accepted aliases:
      - "bicycle"
      - "unicycle"
      - "double-integrator" (also accepts "double_integrator", "double integrator",
        "DoubleIntegrator", etc.)
    """
    s0 = (s or "").strip().lower().replace("_", "-")
    if s0 in {"bicycle"}:
        return "bicycle"
    if s0 in {"unicycle"}:
        return "unicycle"
    if s0 in {"double-integrator", "double integrator"}:
        return "double-integrator"
    raise ValueError(f"Unknown steering_type: {s}")


def _clip(val: float, lo: float, hi: float) -> float:
    """Clamp val into [lo, hi]."""
    return max(lo, min(hi, val))


@dataclass
class Robot:
    """
    Robot(index, steering_type, params=None, state=None)

    Members (kept exactly as requested):
      - kinematics: a callable f(state, u) -> dict of state derivatives
      - index: integer robot id
      - steering_type: one of {"bicycle", "unicycle", "double-integrator"}
      - params: dict with at least {"radius", "max_velocity"}; model-specific
      - state: dict using keys from {x, y, theta, v, v_x, v_y, w, phi};
               each model uses a subset, others are filled with 0.0

    Public methods:
      - step(u, dt): forward-Euler integration step (model-specific controls)
      - state_vector(keys): tuple of state values in the given order
      - set_state(**kwargs): update state by keyword assignments
      - to_dict() / from_dict(d)
      - control_hint(): recommended control field names for this model
    """
    index: int
    steering_type: str
    params: Dict[str, float] = field(default_factory=dict)
    state: Dict[str, float] = field(default_factory=dict)

    # Will be set in __post_init__
    kinematics: Callable[[Dict[str, float], Dict[str, float]], Dict[str, float]] = field(
        init=False, repr=False
    )

    # -------------------------------------------------------------------------
    # Construction & validation
    # -------------------------------------------------------------------------
    def __post_init__(self):
        self.steering_type = _normalize_steering_type(self.steering_type)

        # Merge user params over defaults
        defaults = self._default_params(self.steering_type)
        self.params = {**defaults, **(self.params or {})}

        # Normalize and complete state fields
        needed_keys = self._state_keys(self.steering_type)
        s0 = dict(self.state or {})
        for k in needed_keys:
            s0.setdefault(k, 0.0)
        # Fill all possible keys for safer downstream access
        for k in ["x", "y", "theta", "v", "v_x", "v_y", "w", "phi"]:
            s0.setdefault(k, 0.0)
        self.state = s0

        # Bind kinematics function
        if self.steering_type == "bicycle":
            self.kinematics = self._f_bicycle
        elif self.steering_type == "unicycle":
            self.kinematics = self._f_unicycle
        elif self.steering_type == "double-integrator":
            self.kinematics = self._f_double_integrator

    # -------------------------------------------------------------------------
    # Model-specific: required state keys, default params, control hints
    # -------------------------------------------------------------------------
    @staticmethod
    def _state_keys(steering_type: str) -> Tuple[str, ...]:
        if steering_type == "bicycle":
            # Common minimal set for bicycle: x, y, heading, speed, steering angle
            return ("x", "y", "theta", "v", "phi")
        if steering_type == "unicycle":
            # Keep v, w as part of state container as requested
            return ("x", "y", "theta", "v", "w")
        if steering_type == "double-integrator":
            # Planar position + planar velocity
            return ("x", "y", "v_x", "v_y")
        raise ValueError(steering_type)

    @staticmethod
    def _default_params(steering_type: str) -> Dict[str, float]:
        """
        Provide sensible defaults per model.
        Required by the user: radius, max_velocity.
        """
        if steering_type == "bicycle":
            return dict(
                radius=0.5,
                max_velocity=2.0,
                max_accel=3.0,
                wheelbase=2.5,
                max_steer=0.6,        # |phi| <= 0.6 rad
                max_steer_rate=0.8,   # |phi_dot| <= 0.8 rad/s
            )
        if steering_type == "unicycle":
            return dict(
                radius=0.5,
                max_velocity=2.0,
                max_omega=2.0,        # |w| <= 2 rad/s
            )
        if steering_type == "double-integrator":
            return dict(
                radius=0.5,
                max_velocity=2.0,
                max_accel=3.0,
            )
        raise ValueError(steering_type)

    def control_hint(self) -> Tuple[str, ...]:
        """
        Recommended control field names (informational):
          - bicycle: ('a', 'phi_dot')
          - unicycle: ('v_cmd', 'w_cmd')  # will overwrite state['v'], state['w']
          - double-integrator: ('a_x', 'a_y')
        """
        if self.steering_type == "bicycle":
            return ("a", "phi_dot")
        if self.steering_type == "unicycle":
            return ("v_cmd", "w_cmd")
        if self.steering_type == "double-integrator":
            return ("a_x", "a_y")
        return tuple()

    # -------------------------------------------------------------------------
    # Kinematic models (continuous time)
    # -------------------------------------------------------------------------
    def _f_bicycle(self, state: Dict[str, float], u: Dict[str, float]) -> Dict[str, float]:
        """
        Bicycle model:
          State : x, y, theta, v, phi  (if 'w' also exists, we mirror theta_dot into 'w')
          Input : a, phi_dot
          Limits: |v| <= max_velocity, |phi| <= max_steer, |phi_dot| <= max_steer_rate
        """
        x, y = state["x"], state["y"]
        theta, v, phi = state["theta"], state["v"], state["phi"]
        a = float(u.get("a", 0.0))
        phi_dot = float(u.get("phi_dot", 0.0))

        # Params and input clipping
        L = float(self.params["wheelbase"])
        max_v = float(self.params["max_velocity"])
        max_a = float(self.params["max_accel"])
        max_phi = float(self.params["max_steer"])
        max_phi_dot = float(self.params["max_steer_rate"])

        a = _clip(a, -max_a, max_a)
        phi_dot = _clip(phi_dot, -max_phi_dot, max_phi_dot)

        # Use a clipped steering angle for curvature computation (numerical stability)
        phi_eff = _clip(phi, -max_phi, max_phi)

        x_dot = v * math.cos(theta)
        y_dot = v * math.sin(theta)
        theta_dot = (v / L) * math.tan(phi_eff)
        v_dot = a
        phi_dot_out = phi_dot

        out = {"x": x_dot, "y": y_dot, "theta": theta_dot, "v": v_dot, "phi": phi_dot_out}
        # Optional mirror into 'w' if present in state (helps downstream code that expects w)
        if "w" in state:
            out["w"] = theta_dot
        # Enforce speed limit at integration stage (in step())
        return out

    def _f_unicycle(self, state: Dict[str, float], u: Dict[str, float]) -> Dict[str, float]:
        """
        Unicycle model (velocity-controlled):
          State : x, y, theta, v, w   (v,w act as the current velocity container)
          Input : v_cmd, w_cmd        (will be clipped and written back to state['v'], state['w'])
          Limits: |v_cmd| <= max_velocity, |w_cmd| <= max_omega
        """
        theta = state["theta"]
        v_cmd = float(u.get("v_cmd", state.get("v", 0.0)))
        w_cmd = float(u.get("w_cmd", state.get("w", 0.0)))

        v_cmd = _clip(v_cmd, -float(self.params["max_velocity"]), float(self.params["max_velocity"]))
        w_cmd = _clip(w_cmd, -float(self.params["max_omega"]), float(self.params["max_omega"]))

        x_dot = v_cmd * math.cos(theta)
        y_dot = v_cmd * math.sin(theta)
        theta_dot = w_cmd

        # We carry v and w overwrites through special keys consumed by step()
        out = {"x": x_dot, "y": y_dot, "theta": theta_dot, "v": 0.0, "w": 0.0}
        out["_overwrite_v"] = v_cmd
        out["_overwrite_w"] = w_cmd
        return out

    def _f_double_integrator(self, state: Dict[str, float], u: Dict[str, float]) -> Dict[str, float]:
        """
        2D double-integrator:
          State : x, y, v_x, v_y
          Input : a_x, a_y
          Limits: ||a|| <= max_accel (speed will be limited after integration)
        """
        vx = state["v_x"]
        vy = state["v_y"]
        ax = float(u.get("a_x", 0.0))
        ay = float(u.get("a_y", 0.0))

        # Acceleration magnitude clipping
        a_max = float(self.params["max_accel"])
        a_norm = math.hypot(ax, ay)
        if a_norm > a_max and a_norm > 0.0:
            scale = a_max / a_norm
            ax *= scale
            ay *= scale

        return {"x": vx, "y": vy, "v_x": ax, "v_y": ay}

    # -------------------------------------------------------------------------
    # Integration
    # -------------------------------------------------------------------------
    def step(self, u: Dict[str, float], dt: float) -> None:
        """
        Forward-Euler step: state <- state + f(state,u) * dt
        Also applies post-integration constraints/overwrites when needed.
        """
        deriv = self.kinematics(self.state, u)

        # Integrate plain derivatives
        for k, vdot in deriv.items():
            if k.startswith("_"):
                continue
            self.state[k] = self.state.get(k, 0.0) + float(vdot) * dt

        # Model-specific post processing
        if self.steering_type == "bicycle":
            # Speed limit
            self.state["v"] = _clip(self.state["v"], -self.params["max_velocity"], self.params["max_velocity"])
            # Steering limit
            self.state["phi"] = _clip(self.state["phi"], -self.params["max_steer"], self.params["max_steer"])

        elif self.steering_type == "unicycle":
            # Overwrite v, w with the clipped commands used in this step
            v_cmd = deriv.get("_overwrite_v", self.state.get("v", 0.0))
            w_cmd = deriv.get("_overwrite_w", self.state.get("w", 0.0))
            self.state["v"] = v_cmd
            self.state["w"] = w_cmd

        elif self.steering_type == "double-integrator":
            # Speed magnitude limit after velocity update
            vx, vy = self.state["v_x"], self.state["v_y"]
            v_max = float(self.params["max_velocity"])
            vnorm = math.hypot(vx, vy)
            if vnorm > v_max and vnorm > 0.0:
                scale = v_max / vnorm
                self.state["v_x"] = vx * scale
                self.state["v_y"] = vy * scale

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def state_vector(self, keys: Tuple[str, ...]) -> Tuple[float, ...]:
        """Return a tuple of state values in the given order."""
        return tuple(float(self.state.get(k, 0.0)) for k in keys)

    def set_state(self, **kwargs) -> None:
        """Update state by keyword assignments; keys are used verbatim."""
        for k, v in kwargs.items():
            self.state[k] = float(v)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize robot configuration and state to a plain dict."""
        return dict(
            index=int(self.index),
            steering_type=self.steering_type,
            params=dict(self.params),
            state=dict(self.state),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Robot":
        """Construct a Robot from a dict previously produced by to_dict()."""
        return cls(
            index=int(d["index"]),
            steering_type=str(d["steering_type"]),
            params=dict(d.get("params", {})),
            state=dict(d.get("state", {})),
        )
