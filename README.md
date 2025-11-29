# Multi-Robot Formation Game (Work in Progress)

This repository provides a **research prototype** for multi-robot formation control and **game-theoretic trajectory optimization**.  
We model multiple robots on a 2-D plane, generate letter-shaped formations, and solve an **open-loop dynamic game** using an iLQGames/LQRAX-style method.  
A receding-horizon **MPC loop** is available to execute **one step per round** until all agents reach their assigned goals.

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

### 4. Game Solver (`game_solver.py`)
Defines the **multi-agent dynamic game** and computes an **open-loop Nash equilibrium (OLNE)**.

#### `construct_game(self)`
Symbolically defines the game:
\[
\min_{x^i, u^i}\;
J^i = 
\sum_t 
\Bigl(
w_g\|x_t^i - \hat g_i\|^2 
+ w_c\!\!\sum_{j\neq i}\psi(\|x_t^i-x_t^j\|)
+ w_u\|u_t^i\|^2
\Bigr),
\quad 
x_{t+1}^i = f_i(x_t^i,u_t^i)
\]
where  
- \(\hat g_i = P g_i\) is the **assigned target** for robot *i* (currently fixed),  
- \(f_i\) is the corresponding unicycle/bicycle/double-integrator dynamics.

#### `solve_game(self)`
Numerically computes the **OLNE** using an iLQR/LQRAX-style iteration:
1. Linearize each robot’s dynamics \((A_i, B_i)\);
2. Compute loss gradients via JAX autodiff;
3. Solve per-agent LQ subproblems;
4. Update controls and rollout new trajectories.

**Recent improvements:**
- Gradients are **JIT-compiled once** and reused for all iterations (big speedup).
- Rollout and monitor now use **`lax.scan` + `jit`** to reduce Python overhead.
- Added **external warmup** to compile kernels before the loop (`warmup_only=True`).
- Added **construct-once, refresh-fast** mode: static parts are built once; per-round updates only refresh states and controls.
- Added detailed timing logs per iteration.

---

## ⚙️ MPC Integration (`main.py`)

- Implements a **receding-horizon MPC** loop:
  - Solves the game each round, applies **only the first control step**, and repeats.
  - Warm-starts each round by shifting the previous control sequence.
  - Stops when all robots are within a positional tolerance of their goals.
- Example timing log:
- [TIMING] it=... | linearize=... | grads=... | solve=... | update=... | rollout=... | monitor=... | total=...

---

## 🧠 Objective Summary

Each agent minimizes a weighted combination of:
- **Goal-tracking** loss (\(w_g\|x_t^i-\hat g_i\|^2\)),
- **Collision penalty** (softplus-hinged on squared distances),
- **Control effort** (\(w_u\|u_t^i\|^2\)).

The collision term \(\psi(\cdot)\) is smooth and avoids hard constraints for stability.

---

## 🚀 What’s New (Nov 2025)

- ✅ **MPC one-step execution** with warm-start.
- ✅ **Construct-once / refresh-fast** game setup.
- ✅ **JIT-compiled gradients** reused across rounds.
- ✅ **Rollout and monitor** fully JITed.
- ✅ **External warmup** reduces first-iteration cost.
- ✅ **Detailed timing profiling** for each iteration.

---

## ⚙️ Current Results

- Solver converges for simple formations under moderate weights.
- Visualization shows both initial and final formations.
- Stable gradient evaluation and rollout under most conditions.
- Typical iteration timing (CPU): 15–25 ms after warmup.

---

## ⚠️ Known Issues

1. **Occasional NaN/Inf**
 - Occurs when robots get too close or weights are overly aggressive.
 - Mitigations (softplus, hinge on squared distance) are in place but not foolproof.

2. **Occasional Robot Overlap**
 - Current soft collision penalties may allow slight overlaps.
 - Future work: stronger anisotropic penalties, ESDF distances, or explicit constraints (MCP/MPCC).

---

## 🗺️ Roadmap

1. **Joint optimization of \(P\)** (assignment + trajectories).
2. **ESDF-based shape-aware collision modeling.**
3. **GPU acceleration** via JAX JIT and vectorization.
4. **Convergence visualization** and per-agent cost breakdown.

---

## 📈 Visualization Example

| Initial Condition | Terminal Condition |
|--------------------|--------------------|
| ![Initial](./init_condition.png) | ![Terminal](./terminal_condition.png) |

---

## 👥 Contributors
| Name | Contribution |
|------|---------------|
| **Zehao Wang** | Environment & robot generation, solver/MPC implementation, visualization |
| **Tianyu Qiu** | Framework design |
| **Shotaro Nako** | Code development |

---

## 🧾 Changelog (recent)
- Added **MPC loop** with one-step execution and stopping criteria.  
- Added `refresh_for_mpc()` for per-round updates.  
- Cached **JAX gradients** for speedup.  
- Added **external warmup** compilation.  
- JITed rollout and monitor via `lax.scan`.  
- Added detailed **timing logs**.  
- Smoothed collision penalty to avoid NaNs/Instability.
