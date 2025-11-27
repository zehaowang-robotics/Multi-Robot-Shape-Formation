# Multi-Robot Formation Game (Work in Progress)

This repository provides a **research prototype** for multi-robot formation control and **game-theoretic trajectory optimization**.  
The project aims to model multiple robots on a 2-D plane, generate symbolic letter formations, and later formulate the task as an **open-loop dynamic game** solvable via **LQRAX** or **MCP-based** methods.

---

## ✅ Implemented Components

### 1. Environment (`environment.py`)
- Generates **letter-shaped formations**: `U`, `T`, `A`, `S`, `I`, `N`.
- Supports rectangular bounds (`A x ≤ b`) with automatic coordinate scaling.
- Adapts to arbitrary robot counts:
  - Minimal robots render the formation clearly.
  - Extra robots are placed neatly at the lower-right corner.

### 2. Robot Class (`robot.py`)
- Supports three kinematic models:
  - `bicycle`, `unicycle`, `double-integrator`.
- Each robot includes:
  - `index`, `steering_type`, `params` (`radius`, `max_velocity`, etc.),
  - `state` dict with model-dependent keys (`x`, `y`, `theta`, `v`, …).
- Provides model-specific integration via `step()`.

### 3. Visualization (`basic_utils.py`)
- Core function `visualize_scene()` renders:
  - Robots (blue filled circles with index labels),
  - Goal points (red × markers),
  - Heading / velocity arrows,
  - Scene bounds rectangle.
- Legend placed **outside the upper-right corner** for clarity.

### 4. Main Script (`main.py`)
Integrates all components:
- Creates a `U`-formation environment.
- Samples 10 non-overlapping robot initial poses.
- Visualizes the full setup for verification.

---

## 🧠 Work in Progress

### GameSolver (`game_solver.py`)
Defines structure for a **multi-agent dynamic game solver**.

#### `construct_game(self)`
To symbolically define the game:
\[
\min_{x^i, u^i}\;
J^i = \sum_t \|x_t^i - (E^i P g)\|^2 
     + w_1 \sum_{j\neq i}\mathbf{1}(\|x_t^i-x_t^j\|\le r)
     + w_2 \|u_t^i\|^2,
\quad 
x_{t+1}^i = f(x_t^i,u_t^i)
\]
Result: an MCP/KKT-based formulation suitable for solvers.

#### `solve_game(self)`
To numerically compute the open-loop Nash equilibrium:
- Use **LQRAX**, **PATH**, or custom MCP solver.
- Return equilibrium trajectories \((x^*,u^*)\) and multipliers.
- Support visualization and analysis.

---

## 🧩 Next Steps

1. Finalize symbolic formulation in `construct_game()`.
2. Implement solver logic in `solve_game()` (LQRAX / MCP).
3. Visualize equilibrium trajectories and formation convergence.
4. Extend tests to other formations (`A`, `T`, `S`, …).

---

## 👥 Contributors
| Name | Contribution |
|------|---------------|
| **Zehao Wang** | Environment & robot generation, visualization |
| **Tianyu Qiu** | Framework design |
| **Shotaro Nako** | Code development |

---

## 📅 Project Status
- ✅ Core framework & visualization — **Completed**  
- 🚧 Game formulation & solver — **In progress**  
- 🔜 Equilibrium visualization — **Upcoming**

# Multi-Robot Formation Game (Work in Progress)

This repository provides a **research prototype** for multi-robot formation control and **game-theoretic trajectory optimization**.  
The project aims to model multiple robots on a 2-D plane, generate symbolic letter formations, and later formulate the task as an **open-loop dynamic game** solvable via **LQRAX** or **MCP-based** methods.

---

## ✅ Implemented Components

### 1. Environment (`environment.py`)
- Generates **letter-shaped formations**: `U`, `T`, `A`, `S`, `I`, `N`.
- Supports rectangular bounds (`A x ≤ b`) with automatic coordinate scaling.
- Adapts to arbitrary robot counts:
  - Minimal robots render the formation clearly.
  - Extra robots are placed neatly at the lower-right corner.

### 2. Robot Class (`robot.py`)
- Supports three kinematic models:
  - `bicycle`, `unicycle`, `double-integrator`.
- Each robot includes:
  - `index`, `steering_type`, `params` (`radius`, `max_velocity`, etc.),
  - `state` dict with model-dependent keys (`x`, `y`, `theta`, `v`, …).
- Provides model-specific integration via `step()`.

### 3. Visualization (`basic_utils.py`)
- Core function `visualize_scene()` renders:
  - Robots (blue filled circles with index labels),
  - Goal points (red × markers),
  - Heading / velocity arrows,
  - Scene bounds rectangle.
- Legend placed **outside the upper-right corner** for clarity.

### 4. Main Script (`main.py`)
Integrates all components:
- Creates a `U`-formation environment.
- Samples 10 non-overlapping robot initial poses.
- Visualizes the full setup for verification.

---

## 🧠 Work in Progress

### GameSolver (`game_solver.py`)
Defines structure for a **multi-agent dynamic game solver**.

#### `construct_game(self)`
To symbolically define the game:
\[
\min_{x^i, u^i}\;
J^i = \sum_t \|x_t^i - (E^i P g)\|^2 
     + w_1 \sum_{j\neq i}\mathbf{1}(\|x_t^i-x_t^j\|\le r)
     + w_2 \|u_t^i\|^2,
\quad 
x_{t+1}^i = f(x_t^i,u_t^i)
\]
Result: an MCP/KKT-based formulation suitable for solvers.

#### `solve_game(self)`
To numerically compute the open-loop Nash equilibrium:
- Use **LQRAX**, **PATH**, or custom MCP solver.
- Return equilibrium trajectories \((x^*,u^*)\) and multipliers.
- Support visualization and analysis.

---

## ⚠️ Current Results & Known Issues

- **Results are not yet correct**: the current iLQGames-style loop is still under construction; trajectories and assignments may not converge to the expected formation.
- **Performance**: the prototype uses list-structured rollouts and per-iteration Python loops; vectorization/JIT fusion is pending, so runtime can be slow.
- **API alignment**: the `lqrax` integration has been partially wired; remaining pieces include consistent linearization calls and stable line-search/damping.
- **Collision avoidance**: soft constraints are in place but require tuning (weights/temperature) and optional ESDF-based distances for arbitrary shapes.

We will address these in upcoming commits.

---

## 🧩 Next Steps

1. Finalize symbolic formulation in `construct_game()`.
2. Implement solver logic in `solve_game()` (LQRAX / MCP) with proper linearization, line-search, and damping.
3. Vectorize rollout and loss (JAX `vmap`/`lax.scan`), JIT the full loop, and reduce retracing.
4. Replace Euclidean distance with ESDF for arbitrary shapes; add cutoff masks for pairwise costs.
5. Visualize equilibrium trajectories and validate convergence across formations (`A`, `T`, `S`, …).

---

## 👥 Contributors
| Name | Contribution |
|------|---------------|
| **Zehao Wang** | Environment & robot generation, visualization |
| **Tianyu Qiu** | Framework design |
| **Shotaro Nako** | Code development |

---

## 📅 Project Status
- ✅ Core framework & visualization — **Completed**  
- 🚧 Game formulation & solver — **In progress**  
- 🔜 Equilibrium visualization — **Upcoming**

