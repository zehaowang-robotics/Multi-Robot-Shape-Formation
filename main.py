# main.py
from __future__ import annotations

import argparse
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

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-robot shape formation simulation using game-theoretic solver"
    )
    
    # Define all arguments in a single data structure
    arguments = [
        {"name": ["--assignment-method"], "type": str, "default": "hungarian", "choices": ["greedy", "hungarian", "fair"], "help": "Goal assignment method: greedy, hungarian, or fair (default: hungarian)"},
        {"name": ["--formation"], "type": str, "default": "U", "choices": ["U", "T", "A", "S", "I", "N"], "help": "Formation shape (default: U)"},
        {"name": ["--num-robot"], "type": int, "default": 10, "help": "Number of robots (default: 10)"},
        {"name": ["--xmin", "--x-min"], "type": float, "default": -8.0, "dest": "xmin", "help": "Minimum x coordinate of environment (default: -8.0)"},
        {"name": ["--ymin", "--y-min"], "type": float, "default": -5.0, "dest": "ymin", "help": "Minimum y coordinate of environment (default: -5.0)"},
        {"name": ["--xmax", "--x-max"], "type": float, "default": 8.0, "dest": "xmax", "help": "Maximum x coordinate of environment (default: 8.0)"},
        {"name": ["--ymax", "--y-max"], "type": float, "default": 5.0, "dest": "ymax", "help": "Maximum y coordinate of environment (default: 5.0)"},
        {"name": ["--dt"], "type": float, "default": 0.1, "help": "Time step (default: 0.1)"},
        {"name": ["--T-horizon"], "type": int, "default": 15, "dest": "T_horizon", "help": "MPC time horizon (default: 15)"},
        {"name": ["--pos-tol"], "type": float, "default": 0.25, "dest": "pos_tol", "help": "Position tolerance to declare goal reached (default: 0.25)"},
        {"name": ["--max-mpc-steps"], "type": int, "default": 300, "dest": "max_mpc_steps", "help": "Maximum MPC steps (default: 300)"},
        {"name": ["--vis-every"], "type": int, "default": 1, "dest": "vis_every", "help": "Visualize every N steps (0 to disable, default: 1)"},
        {"name": ["--num-iters"], "type": int, "default": 20, "dest": "num_iters", "help": "Number of inner solver iterations (default: 20)"},
        {"name": ["--step-size"], "type": float, "default": 1e-3, "dest": "step_size", "help": "Solver step size (default: 1e-3)"},
        {"name": ["--w-goal"], "type": float, "default": 1.0, "dest": "w_goal", "help": "Weight for goal tracking (default: 1.0)"},
        {"name": ["--w-collision"], "type": float, "default": 1e3, "dest": "w_collision", "help": "Weight for collision avoidance (default: 1e3)"},
        {"name": ["--w-control"], "type": float, "default": 0.01, "dest": "w_control", "help": "Weight for control effort (default: 0.01)"},
        {"name": ["--w-terminal"], "type": float, "default": 1.0, "dest": "w_terminal", "help": "Weight for terminal cost (default: 1.0)"},
        {"name": ["--radius"], "type": float, "default": 0.40, "dest": "radius_val", "help": "Robot radius (default: 0.40)"},
        {"name": ["--max-velocity"], "type": float, "default": 2.0, "dest": "max_velocity", "help": "Maximum robot velocity (default: 2.0)"},
        {"name": ["--max-omega"], "type": float, "default": 2.0, "dest": "max_omega", "help": "Maximum robot angular velocity (default: 2.0)"},
        {"name": ["--seed"], "type": int, "default": 12345, "help": "Random seed (default: 12345)"},
        {"name": ["--output-prefix"], "type": str, "default": "", "dest": "output_prefix", "help": "Prefix for output filenames (default: empty)"},
    ]
    
    # Add all arguments using a single pattern
    for arg in arguments:
        kwargs = {k: v for k, v in arg.items() if k != "name"}
        parser.add_argument(*arg["name"], **kwargs)
    
    return parser.parse_args()


def main(args):
    """
    Main function for multi-robot shape formation simulation.
    
    Args:
        args: Parsed command-line arguments from argparse
    """
    # Build weights dictionary
    weights = {
        "w_goal": args.w_goal,
        "w_collision": args.w_collision,
        "w_control": args.w_control,
        "w_terminal": args.w_terminal,
    }
    
    # Reproducibility
    rng = np.random.default_rng(args.seed)

    # 1) Environment and goals
    env = Environment(formation=args.formation, num_robot=args.num_robot)
    env.set_rect_bounds(xmin=args.xmin, ymin=args.ymin, xmax=args.xmax, ymax=args.ymax)

    # 2) Sample robots
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

    print("[info] Sampled initial robot states:")

    # 3) Initialize GameSolver params (we will reuse and update per MPC step)
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

    # 5) Visualize initial scene
    print("[info] Visualizing initial scene...")
    visualize_scene(robots, env, filename=f"{args.output_prefix}scene_initial_{args.assignment_method}.png")

    # Initialize list to collect robot snapshots for animation
    robot_snapshots = [copy.deepcopy(robots)]

    # 6) Outer MPC loop
    last_controls = None  # for warm start
    gs.construct_game()
    gs.solve_game(warmup_only=True, do_warmup=True)
    time_start = time.time()
    for k in range(args.max_mpc_steps):
        # shift warm start
        if last_controls is not None:
            u_warm = _shift_warmstart(last_controls)
        else:
            u_warm = None

        # (b) Refresh game model with current robot states
        gs.refresh_for_mpc(u_init_list=u_warm, recompute_assignment=False)

        # (c) Solve current finite-horizon game
        gs.solve_game(do_warmup=False)

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
        # Check if all agents are at the formation goals (what's shown in visualization)
        # The formation is complete when all agents are within tolerance of any formation goal
        pos = np.array([[rb.state["x"], rb.state["y"]] for rb in robots])
        g_all = np.array(env.goals)  # All formation goals
        
        # For each agent, find distance to closest formation goal
        dists = []
        for i in range(N):
            dists_to_goals = np.linalg.norm(g_all - pos[i], axis=1)
            min_dist = np.min(dists_to_goals)
            dists.append(min_dist)
        
        dists = np.array(dists)
        all_reached = bool(np.all(dists <= args.pos_tol))

        print(f"[MPC] step={k:03d}  max_dist={dists.max():.3f}  mean_dist={dists.mean():.3f}  reached={all_reached}")
        
        # (f) Optional intermediate visualization and snapshot collection
        if args.vis_every and (k % args.vis_every == 0):
            visualize_scene(robots, env, filename=f"{args.output_prefix}scene_step_{k}_{args.assignment_method}.png")
            # Collect snapshot for animation
            robot_snapshots.append(copy.deepcopy(robots))

        if all_reached:
            print(f"[MPC] All agents reached goals within tolerance {args.pos_tol}. Terminating at step {k}.")
            break
    time_end = time.time()
    print(f"[info] MPC loop took {time_end - time_start:.2f} seconds.")

    # 7) Final visualization
    print("[info] Visualizing final scene...")
    robots_final = copy.deepcopy(robots)
    visualize_scene(robots_final, env, filename=f"{args.output_prefix}scene_final_{args.assignment_method}.png")
    
    # Add final snapshot to animation (always add to ensure final state is included)
    robot_snapshots.append(robots_final)

    # 8) Create animation from collected snapshots
    if len(robot_snapshots) > 1:
        print("[info] Creating animation from collected snapshots...")
        visualize_scene_animation(
            robot_snapshots, 
            env, 
            filename=f"{args.output_prefix}formation_animation_{args.assignment_method}.gif",
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
    args = parse_args()
    main(args)
