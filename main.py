# main.py
from __future__ import annotations

import numpy as np

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

    # 3) Instantiate the (placeholder) game solver
    gs = GameSolver(
        params={
            "N": N,
            "dt": 0.1,
            "T": 20,
            "weights": {"w_goal": 1.0, "w_collision": 10.0, "w_control": 0.1},
            "radius": radius_val,
        }
    )
    # gs.construct_game()  # placeholder
    # gs.solve_game()      # placeholder

    # 4) Visualize the scene (robots, headings, goals, bounds)
    visualize_scene(robots, env)


if __name__ == "__main__":
    main()