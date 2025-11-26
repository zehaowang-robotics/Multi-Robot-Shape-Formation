# Multi-Robot Formation Game (Work in Progress)

This repository implements the basic framework for a **multi-robot formation control** and **game-theoretic optimization** project.  
The current goal is to model multiple robots on a 2-D plane, visualize symbolic letter formations, and later extend the setup into an **open-loop dynamic game** formulation solvable via **LQRAX** or **MCP-based** methods.

---

## ✅ What Has Been Implemented

### 1. Environment
- Added `Environment` class (`environment.py`) that generates **letter-shaped formations** (`U`, `T`, `A`, `S`, `I`, `N`).
- Supports customizable rectangular bounds (`A x ≤ b`) and automatically scales formation coordinates to fit.
- Supports flexible robot count:
  - Uses the minimal number of robots to render the letter shape clearly.
  - Places any extra robots (if `num_robot > minimal`) neatly in the bottom-right corner of the map.

### 2. Robot Class
- Implemented in `robot.py`.
- Supports multiple **kinematic types**:
  - `bicycle`, `unicycle`, and `double-integrator`.
- Each robot stores:
  - `index`, `steering_type`, `params` (e.g., `radius`, `max_velocity`, etc.)
  - `state` dictionary (`x`, `y`, `theta`, `v`, etc.) depending on the steering type.

### 3. Visualization
- Implemented in `basic.py` with a single function `visualize_scene()`.
- Displays:
  - Robots as blue filled circles with index labels.
  - Goal positions as red × markers.
  - Robot heading and velocity arrows.
  - Rectangular bounds of the environment.
- The legend (robots/goals) is placed **outside the top-right corner** of the figure for clarity.

### 4. Main Script
- Implemented `main.py` to integrate all components:
  - Creates an environment with formation `'U'`.
  - Spawns 10 robots with random initial positions **ensuring no overlap** (distance > sum of radii).
  - Visualizes the full setup for inspection.

---

## 🧠 To Be Completed

### `GameSolver` class (`game_solver.py`)
The structure is defined but not yet implemented.

#### `construct_game(self)`
- To symbolically define the multi-agent open-loop dynamic game:
  - Each agent minimizes  
    \[
    J^i = \sum_t \|x_t^i - (E^i P g)\|^2 + w_1 \sum_{j \neq i}\mathbf{1}(\|x_t^i - x_t^j\| \le r) + w_2 \|u_t^i\|^2
    \]
  - Subject to system dynamics \(x_{t+1}^i = f(x_t^i, u_t^i)\).
- The result will be an **MCP/KKT-based game formulation** ready for the solver.

#### `solve_game(self)`
- To implement the numerical solution procedure:
  - Call the **LQRAX** or **PATH** solver to compute equilibrium trajectories \((x^*, u^*)\).
  - Retrieve and store equilibrium states, controls, and multipliers.
  - Provide an interface for visualization of results.

---

## 🧩 Next Steps

1. Implement the symbolic game structure in `construct_game()`.
2. Add a numerical solver in `solve_game()` (LQRAX, PATH, or custom MCP).
3. Verify convergence to formation goals and visualize trajectories.
4. Extend to other formations (`A`, `T`, `S`, etc.) for testing.

---

## 👥 Contributors

- **Zehao Wang** — Environment and robot generation, visualization.  
- **Tianyu Qiu** — Framework design
- **Shotaro Nako** — Coding

---

## 📅 Status

- Core framework and visualization: **Completed** ✅  
- Game formulation and solver integration: **In progress** 🚧  
- Final equilibrium visualization and testing: **Pending** 🔜
