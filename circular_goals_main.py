# main_circle.py
from __future__ import annotations

import numpy as np
import copy
from robot import Robot
from environment import Environment, generate_circle_goals
from basic_utils import visualize_scene, _bounds_rect_from_Ab, _sample_nonoverlapping_poses
from game_solver import GameSolver


def _x_from_robot(rb: Robot):
    """Minimal state vector used by GameSolver/iLQR per steering_type."""
    st = rb.state
    if rb.steering_type in ("unicycle", "bicycle"):
        return np.array([st["x"], st["y"], st["theta"]], dtype=float)
    elif rb.steering_type == "double-integrator":
        return np.array([st["x"], st["y"], st["v_x"], st["v_y"]], dtype=float)
    else:
        raise ValueError(f"Unknown steering_type {rb.steering_type}")


def _apply_state_to_robot(rb: Robot, x_minimal):
    """Write minimal state back into Robot.state."""
    if rb.steering_type in ("unicycle", "bicycle"):
        rb.set_state(x=float(x_minimal[0]), y=float(x_minimal[1]), theta=float(x_minimal[2]))
    elif rb.steering_type == "double-integrator":
        rb.set_state(x=float(x_minimal[0]), y=float(x_minimal[1]))
        rb.state["v_x"] = float(x_minimal[2])
        rb.state["v_y"] = float(x_minimal[3])
    else:
        raise ValueError(f"Unknown steering_type {rb.steering_type}")


def _shift_warmstart(u_traj_list):
    """Shift control sequences left by one step for warm start; pad last step with zeros."""
    out = []
    for U in u_traj_list:
        U = np.asarray(U)
        if U.shape[0] <= 1:
            out.append(np.zeros_like(U))
        else:
            pad = np.zeros((1, U.shape[1]), dtype=U.dtype)
            out.append(np.vstack([U[1:], pad]))
    return out


def main():
    # Reproducibility
    rng = np.random.default_rng(12345)

    # 1) Environment and circular goals
    #    Use a slightly larger rectangle to comfortably host a 20-agent ring.
    env = Environment(formation="U", num_robot=20)
    env.set_rect_bounds(xmin=-10.0, ymin=-6.0, xmax=10.0, ymax=6.0)

    # 2) Build robots (initial states sampled non-overlapping)
    N = 20
    xmin, ymin, xmax, ymax = _bounds_rect_from_Ab(env.bounds)
    radius_val = 0.35  # robot radius used both for sampling and circle spacing
    radii = [radius_val for _ in range(N)]
    sep_margin = 0.05

    xs, ys, thetas = _sample_nonoverlapping_poses(
        n=N,
        rect=(xmin, ymin, xmax, ymax),
        radii=radii,
        rng=rng,
        max_trials_per_robot=12000,
        angle_uniform=True,
        margin=sep_margin,
    )

    robots = []
    for i in range(N):
        r = Robot(
            index=i,
            steering_type="unicycle",
            params={"radius": radii[i], "max_velocity": 2.0, "max_omega": 2.0},
            state={"x": float(xs[i]), "y": float(ys[i]), "theta": float(thetas[i]), "v": 0.0, "w": 0.0},
        )
        robots.append(r)

    # 3) Generate circle goals using the helper you added in environment.py
    #    The ring radius is auto-chosen to satisfy both neighbor spacing and wall clearance.
    goals_ring = generate_circle_goals(
        num_robot=N,
        robot_radius=radius_val,
        bounds=env.bounds,
        margin_frac=0.12,   # consistent with _fit_points_to_bounds convention
        angle_offset=0.0,   # rotate the ring if desired
    )
    env.goals = goals_ring

    print("[info] Sampled initial robot states and generated circular goals.")

    # 4) MPC parameters
    dt = 0.1
    T_horizon = 20
    pos_tol = 0.10
    max_mpc_steps = 300
    vis_every = 1  # set >0 to visualize every k steps (0 disables intermediate viz)

    # 5) Initialize GameSolver params
    gs = GameSolver(
        params={
            "N": N,
            "dt": dt,
            "T": T_horizon,
            "weights": {"w_goal": 1.0, "w_collision": 1e3, "w_control": 0.01, "w_terminal": 1.0},
            "radius": radius_val,
            "environment": env,
            "robots": robots,
            "g": np.asarray(env.goals, dtype=float),
            "num_iters": 20,
            "step_size": 1e-3,
        }
    )

    # 6) Visualize initial scene
    print("[info] Visualizing initial scene...")
    visualize_scene(robots, env)

    # 7) Outer MPC/game loop
    last_controls = None
    gs.construct_game()
    gs.solve_game(warmup_only=True, do_warmup=True)

    sol = None
    for k in range(max_mpc_steps):
        # Warm start by shifting previous controls
        u_warm = _shift_warmstart(last_controls) if last_controls is not None else None

        # Refresh model with current robot states and (optionally) recompute assignment
        gs.refresh_for_mpc(u_init_list=u_warm, recompute_assignment=True)

        # Solve current finite-horizon game
        try:
            gs.solve_game(do_warmup=False)
        except RuntimeError as e:
            print(f"[WARN] solve_game failed at MPC step {k}: {e}")
            break

        sol = gs.solution
        x_traj_list = sol["x_traj_list"]
        u_traj_list = sol["u_traj_list"]
        last_controls = [np.array(U) for U in u_traj_list]

        # Apply one-step control to each robot using the dynamics built in the game model
        dyn_list = gs.game_model["dyn_list"]
        x0_list = [_x_from_robot(rb) for rb in robots]
        for i in range(N):
            u0 = np.array(u_traj_list[i][0])       # first control
            x1 = np.array(dyn_list[i](x0_list[i], u0))  # one-step forward
            _apply_state_to_robot(robots[i], x1)

        # Check stopping criterion
        hat_g = np.array(sol["hat_g"])  # (N,2)
        pos = np.array([[rb.state["x"], rb.state["y"]] for rb in robots])
        dists = np.linalg.norm(pos - hat_g, axis=1)
        all_reached = bool(np.all(dists <= pos_tol))

        print(f"[MPC] step={k:03d}  max_dist={dists.max():.3f}  mean_dist={dists.mean():.3f}  reached={all_reached}")

        if vis_every and (k % vis_every == 0):
            visualize_scene(robots, env)

        if all_reached:
            print(f"[MPC] All agents reached goals within tolerance {pos_tol}.")
            break

    # 8) Final visualization and optional prints
    print("[info] Visualizing final scene...")
    robots_final = copy.deepcopy(robots)
    visualize_scene(robots_final, env)

    if sol is not None:
        hat_g = sol.get("hat_g", None)
        P = sol.get("P", None)
        if hat_g is not None:
            print("[info] hat_g (assigned goals) shape:", np.asarray(hat_g).shape)
        if P is not None:
            print("[info] P provided by assignment_model with shape:", np.asarray(P).shape)


if __name__ == "__main__":
    main()