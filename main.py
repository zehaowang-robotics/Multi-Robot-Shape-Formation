# main.py
from __future__ import annotations

import argparse
import numpy as np
import copy
import time
import json
from robot import Robot
from environment import Environment, generate_circle_goals
from basic_utils import (
    visualize_scene,
    visualize_scene_animation,
    _bounds_rect_from_Ab,
    _sample_nonoverlapping_poses,
)
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


def _normalize_sequence(seq: str) -> list[str]:
    """Normalize a raw formation string (e.g., 'UTAUSTIN') into a list of letters, filtering empties."""
    seq = (seq or "").strip().upper()
    return [c for c in seq if c.isalpha()]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-robot shape formation with sequential letter switching (UTAUSTIN)."
    )

    # Existing arguments (kept intact; some defaults adjusted for robustness)
    arguments = [
        {"name": ["--assignment-method"], "type": str, "default": "hungarian",
         "choices": ["greedy", "hungarian", "fair"],
         "help": "Goal assignment method: greedy, hungarian, or fair (default: hungarian)"},
        {"name": ["--formation"], "type": str, "default": "U",
         "choices": ["U", "T", "A", "S", "I", "N", ""],
         "help": "Initial formation letter (will be overridden by sequence if provided)."},
        {"name": ["--goal-type"], "type": str, "default": "letter",
         "choices": ["letter", "circle"], "dest": "goal_type",
         "help": "Goal type: 'letter' or 'circle' (sequence requires 'letter')."},
        {"name": ["--circle-margin"], "type": float, "default": 0.12, "dest": "circle_margin",
         "help": "Margin fraction for circular goals (default: 0.12)"},
        {"name": ["--circle-angle-offset"], "type": float, "default": 0.0, "dest": "circle_angle_offset",
         "help": "Angle offset in radians for circular goals (default: 0.0)"},
        {"name": ["--num-robot"], "type": int, "default": 10, "help": "Number of robots (default: 10)"},
        {"name": ["--xmin", "--x-min"], "type": float, "default": -8.0, "dest": "xmin",
         "help": "Minimum x coordinate of environment (default: -8.0)"},
        {"name": ["--ymin", "--y-min"], "type": float, "default": -5.0, "dest": "ymin",
         "help": "Minimum y coordinate of environment (default: -5.0)"},
        {"name": ["--xmax", "--x-max"], "type": float, "default": 8.0, "dest": "xmax",
         "help": "Maximum x coordinate of environment (default: 8.0)"},
        {"name": ["--ymax", "--y-max"], "type": float, "default": 5.0, "dest": "ymax",
         "help": "Maximum y coordinate of environment (default: 5.0)"},
        {"name": ["--dt"], "type": float, "default": 0.1, "help": "Time step (default: 0.1)"},
        {"name": ["--T-horizon"], "type": int, "default": 20, "dest": "T_horizon",
         "help": "MPC time horizon (default: 20)"},
        {"name": ["--pos-tol"], "type": float, "default": 0.3, "dest": "pos_tol",
         "help": "Position tolerance to declare goal reached (default: 0.3)"},
        {"name": ["--max-mpc-steps"], "type": int, "default": 300, "dest": "max_mpc_steps",
         "help": "Maximum MPC steps (default: 300)"},
        {"name": ["--vis-every"], "type": int, "default": 1, "dest": "vis_every",
         "help": "Visualize every N steps (0 to disable, default: 1)"},
        {"name": ["--save-step-figures"], "action": "store_true", "dest": "save_step_figures",
         "help": "Save individual PNG figures for each step (default: False, only saves GIF)"},
        {"name": ["--num-iters"], "type": int, "default": 20, "dest": "num_iters",
         "help": "Number of inner solver iterations (default: 20)"},
        {"name": ["--step-size"], "type": float, "default": 1e-3, "dest": "step_size",
         "help": "Solver step size (default: 1e-3)"},
        {"name": ["--w-goal"], "type": float, "default": 1.0, "dest": "w_goal",
         "help": "Weight for goal tracking (default: 1.0)"},
        {"name": ["--w-collision"], "type": float, "default": 1e3, "dest": "w_collision",
         "help": "Weight for collision avoidance (default: 1e3)"},
        {"name": ["--w-control"], "type": float, "default": 1e-3, "dest": "w_control",
         "help": "Weight for control effort (default: 1e-3)"},
        {"name": ["--w-terminal"], "type": float, "default": 1.0, "dest": "w_terminal",
         "help": "Weight for terminal cost (default: 1.0)"},
        {"name": ["--radius"], "type": float, "default": 0.25, "dest": "radius_val",
         "help": "Robot radius (default: 0.25)"},
        {"name": ["--max-velocity"], "type": float, "default": 2.0, "dest": "max_velocity",
         "help": "Maximum robot velocity (default: 2.0)"},
        {"name": ["--max-omega"], "type": float, "default": 2.0, "dest": "max_omega",
         "help": "Maximum robot angular velocity (default: 2.0)"},
        {"name": ["--seed"], "type": int, "default": 12345, "help": "Random seed (default: 12345)"},
        {"name": ["--output-prefix"], "type": str, "default": "", "dest": "output_prefix",
         "help": "Prefix for output filenames (default: empty)"},
    ]

    # New arguments for sequential formation switching
    arguments += [
        {"name": ["--formation-seq"], "type": str, "default": "UTAUSTIN",
         "help": "Sequential letter sequence to follow. Default: UTAUSTIN"},
        {"name": ["--switch-on"], "type": str, "default": "reach",
         "choices": ["reach", "interval"],
         "help": "Switch policy: 'reach' = when all robots reach current goals; "
                 "'interval' = every --switch-every steps."},
        {"name": ["--switch-every"], "type": int, "default": 30,
         "help": "If --switch-on=interval, switch every K MPC steps (default: 30)."},
        {"name": ["--loop-formation"], "action": "store_true",
         "help": "If set, loop the sequence after finishing the last letter."},
        {"name": ["--random-start-in-seq"], "action": "store_true",
         "help": "If set, start from a random letter inside --formation-seq."},
    ]

    for arg in arguments:
        kwargs = {k: v for k, v in arg.items() if k != "name"}
        parser.add_argument(*arg["name"], **kwargs)

    return parser.parse_args()


def _update_goals_in_solver(gs: GameSolver, env: Environment):
    """Safely propagate new goals to the solver."""
    # Prefer a dedicated API if it exists; else update params directly.
    if hasattr(gs, "update_goals"):
        try:
            gs.update_goals(np.asarray(env.goals, dtype=float))
            return
        except Exception:
            pass
    # Minimal safe update path:
    gs.params["environment"] = env
    gs.params["g"] = np.asarray(env.goals, dtype=float)


def main(args):
    """
    Main function: randomize initial robots, then either:
    - For letter goals: sequentially switch formations through a specified letter sequence (e.g., UTAUSTIN).
    - For circle goals: form the circle and end (no switching).
    """
    # Build weights
    weights = {
        "w_goal": args.w_goal,
        "w_collision": args.w_collision,
        "w_control": args.w_control,
        "w_terminal": args.w_terminal,
    }

    rng = np.random.default_rng(args.seed)

    # === 1) Environment and initial formation ===
    env = Environment(formation=args.formation, num_robot=args.num_robot)
    env.set_rect_bounds(xmin=args.xmin, ymin=args.ymin, xmax=args.xmax, ymax=args.ymax)

    # Handle circle goals separately (no sequence switching)
    if args.goal_type == "circle":
        # Generate circular ring goals
        goals_ring = generate_circle_goals(
            num_robot=args.num_robot,
            robot_radius=args.radius_val,
            bounds=env.bounds,
            margin_frac=args.circle_margin,
            angle_offset=args.circle_angle_offset,
        )
        env.goals = goals_ring
        print(f"[info] Generated {len(goals_ring)} circular ring goals (no sequence switching).")
        # Skip sequence logic for circles
        seq_letters = None
        seq_len = 0
        seq_idx = 0
    else:
        # Letter goals: set up sequence
        seq_letters = _normalize_sequence(args.formation_seq) or ["U"]
        seq_len = len(seq_letters)
        if args.random_start_in_seq:
            seq_idx = int(rng.integers(0, seq_len))
        else:
            seq_idx = 0  # start from first letter of sequence

        # Force environment to the chosen initial letter (regenerate goals)
        env.change_formation(seq_letters[seq_idx], num_robot=args.num_robot)
        print(f"[info] Initial formation set to: {env.formation} (seq index {seq_idx}/{seq_len-1})")

    # === 2) Randomly sample initial robot poses (non-overlapping circles) ===
    N = args.num_robot
    xmin_env, ymin_env, xmax_env, ymax_env = _bounds_rect_from_Ab(env.bounds)
    radii = [args.radius_val for _ in range(N)]
    sep_margin = 0.05

    xs, ys, thetas = _sample_nonoverlapping_poses(
        n=N,
        rect=(xmin_env, ymin_env, xmax_env, ymax_env),
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
            params={"radius": radii[i], "max_velocity": args.max_velocity, "max_omega": args.max_omega},
            state={"x": float(xs[i]), "y": float(ys[i]), "theta": float(thetas[i]), "v": 0.0, "w": 0.0},
        )
        robots.append(r)

    print("[info] Sampled initial robot states.")

    # === 3) Initialize GameSolver ===
    gs = GameSolver(
        params={
            "N": N,
            "dt": args.dt,
            "T": args.T_horizon,
            "weights": weights,
            "radius": args.radius_val,
            "environment": env,
            "robots": robots,
            "g": np.asarray(env.goals, dtype=float),
            "num_iters": args.num_iters,
            "step_size": args.step_size,
            "assignment_method": args.assignment_method,
        }
    )

    # === 4) Initial visualization ===
    if args.save_step_figures:
        print("[info] Visualizing initial scene...")
        if args.goal_type == "circle":
            filename = f"{args.output_prefix}scene_initial_circle_N{args.num_robot}.png"
        else:
            filename = f"{args.output_prefix}scene_initial_seq_{args.formation_seq}_N{args.num_robot}.png"
        visualize_scene(robots, env, filename=filename)
    else:
        print("[info] Preparing animation...")

    # Collect snapshots for GIF
    robot_snapshots = [copy.deepcopy(robots)]
    goals_snapshots = [np.array(env.goals, dtype=float)]
    if args.goal_type == "letter":
        formation_labels = [env.formation]
    else:
        formation_labels = ["circle"]

    # === 5) Warmup once (if solver supports it) ===
    gs.construct_game()
    gs.solve_game(warmup_only=True, do_warmup=True)

    # === 6) Outer MPC loop with sequential formation switching ===
    last_controls = None
    time_start = time.time()

    # Book-keeping for sequence finish (only for letter goals)
    finished_once = False
    force_reassign_for_next = False
    
    # Trajectory logging: store actual executed trajectories
    # Format: trajectories[agent_id][step] = {"x": float, "y": float, "theta": float, ...}
    trajectories = {i: [] for i in range(N)}
    
    # Log initial state (before MPC loop)
    for i in range(N):
        rb = robots[i]
        state_dict = {
            "step": -1,  # -1 indicates initial state
            "x": float(rb.state["x"]),
            "y": float(rb.state["y"]),
            "theta": float(rb.state.get("theta", 0.0)),
        }
        if "v" in rb.state:
            state_dict["v"] = float(rb.state["v"])
        if "w" in rb.state:
            state_dict["w"] = float(rb.state["w"])
        if "v_x" in rb.state:
            state_dict["v_x"] = float(rb.state["v_x"])
        if "v_y" in rb.state:
            state_dict["v_y"] = float(rb.state["v_y"])
        trajectories[i].append(state_dict)
    
    for k in range(args.max_mpc_steps):
        # Warm start controls
        u_warm = _shift_warmstart(last_controls) if last_controls is not None else None

        # Refresh model for current states
        gs.refresh_for_mpc(u_init_list=u_warm, recompute_assignment=force_reassign_for_next)
        force_reassign_for_next = False

        # Solve finite-horizon game
        gs.solve_game(do_warmup=False)
        sol = gs.solution
        x_traj_list = sol["x_traj_list"]
        u_traj_list = sol["u_traj_list"]
        last_controls = [np.array(U) for U in u_traj_list]

        # Apply one step to real robots
        dyn_list = gs.game_model["dyn_list"]
        x0_list = [_x_from_robot(rb) for rb in robots]
        for i in range(N):
            u0 = np.array(u_traj_list[i][0])
            x1 = np.array(dyn_list[i](x0_list[i], u0))
            _apply_state_to_robot(robots[i], x1)
            
            # Log actual trajectory state
            rb = robots[i]
            state_dict = {
                "step": k,
                "x": float(rb.state["x"]),
                "y": float(rb.state["y"]),
                "theta": float(rb.state.get("theta", 0.0)),
            }
            # Add velocity information if available
            if "v" in rb.state:
                state_dict["v"] = float(rb.state["v"])
            if "w" in rb.state:
                state_dict["w"] = float(rb.state["w"])
            if "v_x" in rb.state:
                state_dict["v_x"] = float(rb.state["v_x"])
            if "v_y" in rb.state:
                state_dict["v_y"] = float(rb.state["v_y"])
            trajectories[i].append(state_dict)

        # Check distance-to-goal
        pos = np.array([[rb.state["x"], rb.state["y"]] for rb in robots])
        g_all = np.array(env.goals)
        dists = np.linalg.norm(pos[:, None, :] - g_all[None, :, :], axis=2).min(axis=1)
        all_reached = bool(np.all(dists <= args.pos_tol))

        print(f"[MPC] step={k:03d}  max_dist={dists.max():.3f}  mean_dist={dists.mean():.3f}  reached={all_reached}")

        # Visualization & snapshot
        if args.vis_every and (k % args.vis_every == 0):
            if args.save_step_figures:
                if args.goal_type == "circle":
                    filename = f"{args.output_prefix}scene_step_{k}_circle_N{args.num_robot}.png"
                else:
                    filename = f"{args.output_prefix}scene_step_{k}_{env.formation}_N{args.num_robot}.png"
                visualize_scene(robots, env, filename=filename)
            robot_snapshots.append(copy.deepcopy(robots))
            goals_snapshots.append(np.array(env.goals, dtype=float))
            if args.goal_type == "letter":
                formation_labels.append(env.formation)
            else:
                formation_labels.append("circle")

        # === Switching policy (only for letter goals) ===
        if args.goal_type == "letter":
            need_switch = False
            if args.switch_on == "reach":
                need_switch = all_reached
            elif args.switch_on == "interval":
                need_switch = (k > 0 and (k % args.switch_every == 0))

            if need_switch:
                # Move to next letter in sequence
                next_idx = (seq_idx + 1) % seq_len
                if next_idx == 0 and not args.loop_formation:
                    # Sequence finished once; stop if not looping
                    finished_once = True
                else:
                    seq_idx = next_idx

                if finished_once:
                    print(f"[MPC] Sequence '{args.formation_seq}' completed. Terminating.")
                    break

                # Change environment formation & goals
                next_letter = seq_letters[seq_idx]
                env.change_formation(next_letter, num_robot=N)
                print(f"[info] Switched formation -> {env.formation} (seq index {seq_idx}/{seq_len-1})")

                # Push new goals into solver safely
                _update_goals_in_solver(gs, env)
                gs.construct_game()                 # rebuild graph so costs/constraints bind to new goals
                last_controls = None                # drop warm-start after big goal jump
                force_reassign_for_next = True      # force re-run Hungarian/fair/etc. at next refresh

                # Keep a snapshot right after switching (for smoother GIF)
                robot_snapshots.append(copy.deepcopy(robots))
                goals_snapshots.append(np.array(env.goals, dtype=float))
                formation_labels.append(env.formation)

            # Early stop if everything reached and no switching requested
            if all_reached and args.switch_on != "reach":
                print(f"[MPC] All agents reached current goals; no switch policy active. Terminating.")
                break
        else:
            # For circle goals: just end when all agents reach goals
            if all_reached:
                print(f"[MPC] All agents reached circle formation. Terminating.")
                break

    time_end = time.time()
    print(f"[info] MPC loop took {time_end - time_start:.2f} seconds.")

    # Save trajectories to JSON file
    # Convert goals to JSON-serializable format
    initial_goals = [[float(g[0]), float(g[1])] for g in env.goals] if hasattr(env, "goals") and env.goals else []
    final_goals = [[float(g[0]), float(g[1])] for g in env.goals] if hasattr(env, "goals") and env.goals else []
    
    trajectory_data = {
        "metadata": {
            "num_agents": N,
            "num_steps": len(trajectories[0]) if trajectories[0] else 0,
            "goal_type": args.goal_type,
            "assignment_method": args.assignment_method,
            "formation_seq": args.formation_seq if args.goal_type == "letter" else "circle",
            "dt": args.dt,
            "T_horizon": args.T_horizon,
            "pos_tol": args.pos_tol,
            "total_time_seconds": float(time_end - time_start),
        },
        "goals": {
            "initial": initial_goals,
            "final": final_goals,
        },
        "trajectories": trajectories,
    }
    
    # Generate output filename
    if args.goal_type == "circle":
        traj_filename = f"{args.output_prefix}trajectories_{args.assignment_method}_circle_N{args.num_robot}.json"
    else:
        traj_filename = f"{args.output_prefix}trajectories_{args.assignment_method}_{args.formation_seq}_N{args.num_robot}.json"
    
    with open(traj_filename, "w") as f:
        json.dump(trajectory_data, f, indent=2)
    print(f"[info] Saved trajectories to {traj_filename}")

    # Final visualization
    robots_final = copy.deepcopy(robots)
    if args.save_step_figures:
        print("[info] Visualizing final scene...")
        if args.goal_type == "circle":
            filename = f"{args.output_prefix}scene_final_circle_N{args.num_robot}.png"
        else:
            filename = f"{args.output_prefix}scene_final_{env.formation}_N{args.num_robot}.png"
        visualize_scene(robots_final, env, filename=filename)
    robot_snapshots.append(robots_final)
    goals_snapshots.append(np.array(env.goals, dtype=float))
    if args.goal_type == "letter":
        formation_labels.append(env.formation)
    else:
        formation_labels.append("circle")

    # Build GIF
    if len(robot_snapshots) > 1:
        print("[info] Creating animation from collected snapshots...")
        if args.goal_type == "circle":
            filename = f"{args.output_prefix}formation_animation_{args.assignment_method}_circle_N{args.num_robot}.gif"
        else:
            filename = f"{args.output_prefix}formation_animation_{args.assignment_method}_{args.formation_seq}_N{args.num_robot}.gif"
        visualize_scene_animation(
            robot_snapshots,
            env,
            filename=filename,
            duration=0.15,
            step_label=True,
            goals_snapshots=goals_snapshots,          # NEW
            formation_labels=formation_labels         # NEW
        )

    # Optional: print assignment info
    if gs.solution is not None:
        hat_g = gs.solution.get("hat_g", None)
        P = gs.solution.get("P", None)
        if hat_g is not None:
            print("[info] hat_g (assigned goals) shape:", np.asarray(hat_g).shape)
        if P is not None:
            print("[info] P provided by assignment_model with shape:", np.asarray(P).shape)


if __name__ == "__main__":
    args = parse_args()
    main(args)
