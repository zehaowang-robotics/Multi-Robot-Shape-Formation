# main.py
from __future__ import annotations

import numpy as np
import copy
from robot import Robot
from environment import Environment
from basic_utils import visualize_scene, _bounds_rect_from_Ab, _sample_nonoverlapping_poses
from game_solver import GameSolver

def main():
    # Reproducibility
    rng = np.random.default_rng(12345)

    # 1) Environment: larger scene and "U" formation
    env = Environment(formation="U", num_robot=10)
    # Enlarge the rectangle (wider area than default): [-8, 8] x [-5, 5]
    env.set_rect_bounds(xmin=-8.0, ymin=-5.0, xmax=8.0, ymax=5.0)

    # 2) Sample 10 robots with non-overlapping initial states
    N = 10
    xmin, ymin, xmax, ymax = _bounds_rect_from_Ab(env.bounds)

    # Choose per-robot radii (constant here, but could be heterogeneous)
    radius_val = 0.40
    radii = [radius_val for _ in range(N)]

    # Extra margin to be conservative on overlaps
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

    # 3) Instantiate and wire the game solver
    gs = GameSolver(
        params={
            "N": N,
            "dt": 0.1,
            "T": 100,
            "weights": {"w_goal": 1.0, "w_collision": 10.0, "w_control": 0.1, "w_terminal": 1.0},
            "radius": radius_val,
            # 关键：把环境与机器人交给 GameSolver，g(目标)从 env 提供
            "environment": env,
            "robots": robots,
            "g": np.asarray(env.goals, dtype=float),
            # 可选：初始控制不提供则自动置零
            # 可选：若有分配模型或固定P，可加：
            # "assignment_model": your_assignment_model,
            # "P": your_fixed_P,  # (N,N)
            # 可选：迭代参数
            "num_iters": 100,
            "step_size": 1e-4,
        }
    )
    
    print("[info] Starting construct the game...")

    # 4) Build and solve the game
    gs.construct_game()
    
    print("[info] Trying to solve the game...")
    try:
        gs.solve_game()
    except RuntimeError as e:
        print(f"[WARN] solve_game failed: {e}")
        visualize_scene(robots, env)
        return

    # 5) 先画“初始状态图”（此时 robots 仍是初始状态）
    print("[info] Visualizing initial scene...")
    visualize_scene(robots, env)

    # 6) 再画“最终状态图”：用拷贝承载终点解，避免覆盖初始 robots
    sol = gs.solution
    x_traj_list = sol["x_traj_list"]

    robots_final = copy.deepcopy(robots)
    for i in range(N):
        x_T = np.array(x_traj_list[i][-1])  # unicycle/bicycle: [x, y, theta]
        robots_final[i].set_state(x=float(x_T[0]), y=float(x_T[1]), theta=float(x_T[2]))

    print("[info] Visualizing final scene...")
    visualize_scene(robots_final, env)

    # （可选）打印分配信息
    hat_g = sol.get("hat_g", None)
    P = sol.get("P", None)
    if hat_g is not None:
        print("[info] hat_g (assigned goals) shape:", np.asarray(hat_g).shape)
    if P is not None:
        print("[info] P provided by assignment_model with shape:", np.asarray(P).shape)
        
if __name__ == "__main__":
    main()
