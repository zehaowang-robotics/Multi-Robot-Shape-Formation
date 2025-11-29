# main.py
from __future__ import annotations

import numpy as np
import copy
from robot import Robot
from environment import Environment
from basic_utils import visualize_scene, visualize_scene_animation, _bounds_rect_from_Ab, _sample_nonoverlapping_poses
from game_solver import GameSolver
import time

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
        # If Robot supports velocity keys, set them as well.
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

    # 1) Environment and goals
    env = Environment(formation="U", num_robot=10)
    env.set_rect_bounds(xmin=-8.0, ymin=-5.0, xmax=8.0, ymax=5.0)

    # 2) Sample robots
    N = 10
    xmin, ymin, xmax, ymax = _bounds_rect_from_Ab(env.bounds)
    radius_val = 0.40
    radii = [radius_val for _ in range(N)]
    sep_margin = 0.05

    xs, ys, thetas = _sample_nonoverlapping_poses(
        n=N,
        rect=(xmin, ymin, xmax, ymax),
        radii=radii,
        rng=rng,
        max_trials_per_robot=8000,
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

    print("[info] Sampled initial robot states:")

    # 3) MPC parameters
    dt = 0.1
    T_horizon = 15            # short finite horizon for MPC
    pos_tol = 0.25            # position tolerance to declare 'reached'
    max_mpc_steps = 300       # safety cap on outer MPC steps
    vis_every = 0             # set >0 to visualize every k steps (0 disables intermediate viz)

    # 4) Initialize GameSolver params (we will reuse and update per MPC step)
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
            "num_iters": 20,     # fewer inner iters per MPC round is typical
            "step_size": 1e-3,
        }
    )

    # 5) Visualize initial scene
    print("[info] Visualizing initial scene...")
    visualize_scene(robots, env, filename="scene_initial.png")

    # Initialize list to collect robot snapshots for animation
    robot_snapshots = [copy.deepcopy(robots)]

    # 6) Outer MPC loop
    last_controls = None  # for warm start
    gs.construct_game()
    gs.solve_game(warmup_only=True, do_warmup=True)
    time_start = time.time()
    for k in range(max_mpc_steps):
        # shift warm start
        if last_controls is not None:
            u_warm = _shift_warmstart(last_controls)
        else:
            u_warm = None

        # (b) Refresh game model with current robot states
        gs.refresh_for_mpc(u_init_list=u_warm, recompute_assignment=True)

        # (c) Solve current finite-horizon game
        try:
            gs.solve_game(do_warmup=False)
        except RuntimeError as e:
            print(f"[WARN] solve_game failed at MPC step {k}: {e}")
            break

        sol = gs.solution
        x_traj_list = sol["x_traj_list"]
        u_traj_list = sol["u_traj_list"]
        last_controls = [np.array(U) for U in u_traj_list]  # cache for next warm start

        # (d) Apply *one* step of the planned motion to the real robots
        #     We step using dyn_list[ i ] from the built model to ensure consistency.
        dyn_list = gs.game_model["dyn_list"]
        x0_list = [ _x_from_robot(rb) for rb in robots ]
        for i in range(N):
            u0 = np.array(u_traj_list[i][0])       # first control
            x1 = np.array(dyn_list[i](x0_list[i], u0))  # one-step forward
            _apply_state_to_robot(robots[i], x1)

        # (e) Check stopping criterion
        hat_g = np.array(sol["hat_g"])  # (N,2)
        pos = np.array([[rb.state["x"], rb.state["y"]] for rb in robots])
        dists = np.linalg.norm(pos - hat_g, axis=1)
        all_reached = bool(np.all(dists <= pos_tol))

        print(f"[MPC] step={k:03d}  max_dist={dists.max():.3f}  mean_dist={dists.mean():.3f}  reached={all_reached}")

        # (f) Optional intermediate visualization and snapshot collection
        if vis_every and (k % vis_every == 0):
            visualize_scene(robots, env, filename=f"scene_step_{k}.png")
            # Collect snapshot for animation
            robot_snapshots.append(copy.deepcopy(robots))

        if all_reached:
            print(f"[MPC] All agents reached goals within tolerance {pos_tol}.")
            break
    time_end = time.time()
    print(f"[info] MPC loop took {time_end - time_start:.2f} seconds.")

    # 7) Final visualization
    print("[info] Visualizing final scene...")
    robots_final = copy.deepcopy(robots)
    visualize_scene(robots_final, env, filename="scene_final.png")
    
    # Add final snapshot to animation (always add to ensure final state is included)
    robot_snapshots.append(robots_final)

    # 8) Create animation from collected snapshots
    if len(robot_snapshots) > 1:
        print("[info] Creating animation from collected snapshots...")
        visualize_scene_animation(
            robot_snapshots, 
            env, 
            filename="formation_animation.gif",
            duration=0.15,  # frame duration in seconds
            step_label=True
        )

    # (optional) print assignment info from last solve
    if sol is not None:
        hat_g = sol.get("hat_g", None)
        P = sol.get("P", None)
        if hat_g is not None:
            print("[info] hat_g (assigned goals) shape:", np.asarray(hat_g).shape)
        if P is not None:
            print("[info] P provided by assignment_model with shape:", np.asarray(P).shape)

if __name__ == "__main__":
    main()
