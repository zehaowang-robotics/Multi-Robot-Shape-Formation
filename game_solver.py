# game_solver.py
from __future__ import annotations
from time import perf_counter as _now
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from robot import Robot
from environment import Environment
from lqrax import iLQR
import numpy as np
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import jit, vmap, grad, lax
@dataclass
class GameSolver:
    """
    GameSolver
    -----------
    A class for setting up and solving the multi-agent formation game
    using an LQRAX-style analytical solver (Linear-Quadratic Relaxation for Analytical eXpressions).

    Reference formulation (from the provided LaTeX):
        - N robots on a 2-D plane.
        - Each robot i has a goal g^i.
        - The joint assignment matrix P maps the goal set g to
          the assigned goals \hat{g} = P * g.
        - Each agent optimizes its own cost J^i(x, u; P) within the coupled game Γ(P).
        - The equilibrium satisfies all agents' KKT conditions.

    This class defines the parameter container and interface for constructing
    and solving such a game. The detailed numerical formulation will later
    connect to LQRAX or other MCP-based solvers for OLNE (Open-loop Nash Equilibrium).
    """

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Initialize default parameter fields.
        Users can overwrite these values as needed before constructing the game.
        """
        defaults = dict(
            N=4,                 # number of robots
            dt=0.1,              # time step
            T=20,                # time horizon
            weights={
                "w_goal": 1.0,        # weight for goal tracking
                "w_collision": 10.0,  # weight for collision avoidance
                "w_control": 0.1,     # weight for control effort
            },
            radius=0.5,          # safe distance between robots
        )
        # merge user-specified params
        self.params = {**defaults, **self.params}

        # placeholders for future data
        self.game_model: Optional[Any] = None
        self.solution: Optional[Any] = None

    # -------------------------------------------------------------------------
    # Core methods
    # -------------------------------------------------------------------------
    def construct_game(self):
        """
        Construct the game model Γ(P) from parameters, auto-wiring Robot/Environment.

        Expected in self.params (some are optional with autoguess):
            - robots: List[Robot]                                   (recommended)
            - environment: Environment                              (recommended)
            - g: np/jnp array of shape (N,2)                        (optional; else from environment.goals)
            - x0_list: list of N arrays for initial states          (optional; else from robots.state)
            - u_init_list: list of N arrays with shape (T,u_dim)    (optional; zeros if missing)
            - assignment_model: optional, must have hat_g(g)->(N,2) (optional)
            - P: optional numeric (N,N) for fixed assignment        (optional, fallback to greedy NN)

        We build JAX-friendly dynamics per robot (minimal state for iLQGames):
            - bicycle        : x=[x,y,theta], u=[v, delta]  with wheelbase L
            - unicycle       : x=[x,y,theta], u=[v, w]
            - double-integrator: x=[x,y,vx,vy], u=[ax, ay]
        """

        p = self.params
        N = int(p.get("N", 0))
        T = int(p["T"]); dt = float(p["dt"])
        w = p["weights"]; r_safe = float(p["radius"])

        # --- 0) Pull robots/environment if present; otherwise rely on user-provided arrays ---
        robots = p.get("robots", None)
        env = p.get("environment", None)

        # --- 1) Goals g (N,2) ---
        if "g" in p:
            g = jnp.asarray(p["g"])
        else:
            if env is None:
                raise ValueError("params['g'] missing and no environment provided to infer goals.")
            if N == 0:
                N = int(env.num_robot)
            g = jnp.asarray(env.goals, dtype=float)
            if g.shape[0] != N:
                # If env has more points than N, take first N
                g = g[:N, :]
        # store back normalized N
        if N == 0:
            N = int(g.shape[0])
        self.params["N"] = N

        # --- 2) Build JAX dynamics per robot from Robot.steering_type ---
        dyn_list = []
        x_dim_list, u_dim_list = [], []
        x0_list = []

        def _wrap_bicycle(L, v_max):
            # x=[x,y,theta], u=[v, delta]
            def f(x, u):
                theta = x[2]
                v, delta = u[0], u[1]
                v = jnp.clip(v, -v_max, v_max)
                dx = jnp.array([v * jnp.cos(theta),
                                v * jnp.sin(theta),
                                v * jnp.tan(delta) / L])
                return x + dt * dx
            return jit(f)

        def _wrap_unicycle(v_max, w_max):
            # x=[x,y,theta], u=[v, w]
            def f(x, u):
                theta = x[2]
                v = jnp.clip(u[0], -v_max, v_max)
                w = jnp.clip(u[1], -w_max, w_max)
                dx = jnp.array([v * jnp.cos(theta),
                                v * jnp.sin(theta),
                                w])
                return x + dt * dx
            return jit(f)

        def _wrap_double_integrator(a_max, v_max):
            # x=[x,y,vx,vy], u=[ax, ay]
            def f(x, u):
                ax, ay = u[0], u[1]
                a_norm = jnp.linalg.norm(u)
                scale = jnp.minimum(1.0, a_max / (a_norm + 1e-12))
                ax, ay = ax * scale, ay * scale
                x_next = jnp.array([
                    x[0] + dt * x[2],
                    x[1] + dt * x[3],
                    x[2] + dt * ax,
                    x[3] + dt * ay
                ])
                # clip speed
                vxy = x_next[2:4]
                vnorm = jnp.linalg.norm(vxy)
                vxy = jnp.where(vnorm > v_max, vxy * (v_max / (vnorm + 1e-12)), vxy)
                x_next = x_next.at[2:4].set(vxy)
                return x_next
            return jit(f)

        # helper to pull initial state vector from Robot.state, mapped to minimal JAX state
        def _x0_from_robot(rb: "Robot"):
            st = rb.state
            if rb.steering_type == "bicycle" or rb.steering_type == "unicycle":
                return jnp.array([st["x"], st["y"], st["theta"]], dtype=float)
            elif rb.steering_type == "double-integrator":
                return jnp.array([st["x"], st["y"], st["v_x"], st["v_y"]], dtype=float)
            else:
                raise ValueError(f"Unknown steering_type {rb.steering_type}")

        if robots is not None:
            if N == 0:
                N = len(robots)
                self.params["N"] = N
            if len(robots) != N:
                raise ValueError(f"len(robots)={len(robots)} != N={N}")

            for rb in robots:
                if rb.steering_type == "bicycle":
                    L = float(rb.params["wheelbase"])
                    v_max = float(rb.params["max_velocity"])
                    dyn_list.append(_wrap_bicycle(L, v_max))
                    x_dim_list.append(3); u_dim_list.append(2)
                    x0_list.append(_x0_from_robot(rb))
                elif rb.steering_type == "unicycle":
                    v_max = float(rb.params["max_velocity"])
                    w_max = float(rb.params["max_omega"])
                    dyn_list.append(_wrap_unicycle(v_max, w_max))
                    x_dim_list.append(3); u_dim_list.append(2)
                    x0_list.append(_x0_from_robot(rb))
                elif rb.steering_type == "double-integrator":
                    a_max = float(rb.params["max_accel"])
                    v_max = float(rb.params["max_velocity"])
                    dyn_list.append(_wrap_double_integrator(a_max, v_max))
                    x_dim_list.append(4); u_dim_list.append(2)
                    x0_list.append(_x0_from_robot(rb))
                else:
                    raise ValueError(f"Unknown steering_type {rb.steering_type}")
        else:
            # If no Robot list was provided, we expect user has given x0_list AND (x_dim/u_dim) implicitly consistent.
            raise ValueError("params['robots'] must be provided to auto-wire dynamics from Robot.")

        x0_list = p.get("x0_list", x0_list)
        x0_list = [jnp.asarray(x0) for x0 in x0_list]

        # --- 3) Initial controls (zeros by default) ---
        if "u_init_list" in p:
            u_traj_list = [jnp.asarray(U) for U in p["u_init_list"]]
        else:
            u_traj_list = []
            for k in range(N):
                U = jnp.zeros((T, u_dim_list[k]), dtype=float)
                u_traj_list.append(U)
        # keep a copy in params for transparency
        self.params["u_init_list"] = u_traj_list

        # --- 4) Assignment hat_g (N,2): priority = assignment_model > P > greedy NN > identity ---
        if "assignment_model" in p and p["assignment_model"] is not None:
            hat_g = p["assignment_model"].hat_g(g)
        elif "P" in p and p["P"] is not None:
            P = jnp.asarray(p["P"])
            if P.shape != (N, N):
                raise ValueError(f"P must be shape {(N,N)}, got {P.shape}.")
            hat_g = P @ g
        else:
            # greedy nearest-neighbor: match initial positions to goals
            x0_xy = jnp.stack([x0[:2] for x0 in x0_list], axis=0)  # (N,2)
            used = set()
            hat = []
            for i in range(N):
                dists = jnp.linalg.norm(g - x0_xy[i], axis=1)
                order = np.argsort(np.array(dists))  # use numpy for simple argsort
                chosen = None
                for idx in order:
                    if int(idx) not in used:
                        chosen = int(idx); break
                if chosen is None: chosen = int(order[0])
                used.add(chosen)
                hat.append(g[chosen])
            hat_g = jnp.stack(hat, axis=0)  # (N,2)

        # --- 5) Build smooth runtime and terminal losses (JAX-friendly) ---
        def _softplus(x, alpha=20.0):
            return jnn.softplus(alpha * x) / alpha

        def psi_dist(d):
            # smooth (r - d)_+
            return _softplus(r_safe - d)

        def make_runtime_loss(i: int):
            w_goal = float(w.get("w_goal", 1.0))
            w_collision = float(w.get("w_collision", 10.0))
            w_control = float(w.get("w_control", 0.1))

            def safe_norm(x, axis=-1, eps=1e-6):
                # sqrt(sum(x^2) + eps) ：在 0 处梯度为 0，不会 NaN
                return jnp.sqrt(jnp.sum(x * x, axis=axis) + eps)

            def rt(x_i_t, u_i_t, x_all_t):
                # 只堆位置，避免不同 x_dim 的影响
                pos_all = jnp.stack([x[:2] for x in x_all_t], axis=0)  # (N,2)
                x_i_xy = x_i_t[:2]

                goal_term = w_goal * jnp.sum((x_i_xy - hat_g[i]) ** 2)

                # pairwise diffs
                diffs = pos_all - x_i_xy  # (N,2)

                # （可选）把自项差分改成 0，或改成一个与 x_i 独立的常数向量
                diffs = diffs.at[i].set(jnp.array([0.0, 0.0]))

                # 安全范数（含 eps），不会在 0 处产生 NaN 梯度
                dists = safe_norm(diffs, axis=1)  # (N,)

                # mask 自项
                mask = jnp.ones((N,), dtype=dists.dtype).at[i].set(0.0)

                coll_sum = jnp.sum(mask * psi_dist(dists))
                coll_term = w_collision * coll_sum

                ctrl_term = w_control * jnp.sum(u_i_t ** 2)
                return goal_term + coll_term + ctrl_term

            return rt


        runtime_losses = [make_runtime_loss(i) for i in range(N)]

        def terminal_loss(x_T_all):
            wT = float(w.get("w_terminal", 1.0))
            val = 0.0
            for i in range(N):
                val = val + wT * jnp.sum((x_T_all[i][:2] - hat_g[i]) ** 2)
            return val

        # --- 6) Lightweight rollout step functions (discrete already) ---
        def rollout_agent(f_step, x0, U):
            xs = []
            x = x0
            for t in range(T):
                x = f_step(x, U[t])
                xs.append(x)
            return jnp.stack(xs, axis=0)  # (T, x_dim)
        
        ilqr_agents = []

        # 为每个机器人构造 iLQR 子类，按最小状态定义 dyn(xt, ut) 返回连续时间导数
        for i, rb in enumerate(self.params["robots"]):
            stype = rb.steering_type

            if stype == "unicycle":
                v_max = float(rb.params["max_velocity"])
                w_max = float(rb.params["max_omega"])

                class UniAgent(iLQR):
                    def dyn(self, xt, ut):
                        theta = xt[2]
                        v = jnp.clip(ut[0], -v_max, v_max)
                        w = jnp.clip(ut[1], -w_max, w_max)
                        return jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), w])

                ilqr_agents.append(UniAgent(dt=dt, x_dim=3, u_dim=2,
                                            Q=jnp.eye(3)*1e-2, R=jnp.eye(2)*1e-2))

            elif stype == "bicycle":
                L = float(rb.params["wheelbase"])
                v_max = float(rb.params["max_velocity"])

                class BikeAgent(iLQR):
                    def dyn(self, xt, ut):
                        x, y, theta = xt
                        v, delta = ut
                        v = jnp.clip(v, -v_max, v_max)
                        dx = v * jnp.cos(theta)
                        dy = v * jnp.sin(theta)
                        dtheta = v * jnp.tan(delta) / L
                        return jnp.array([dx, dy, dtheta])

                ilqr_agents.append(BikeAgent(dt=dt, x_dim=3, u_dim=2,
                                            Q=jnp.eye(3)*1e-2, R=jnp.eye(2)*1e-2))

            elif stype == "double-integrator":
                a_max = float(rb.params["max_accel"])

                class DI2DAgent(iLQR):
                    def dyn(self, xt, ut):
                        # xt = [x, y, vx, vy], ut = [ax, ay]
                        ax, ay = ut
                        a_norm = jnp.linalg.norm(ut) + 1e-12
                        scale = jnp.minimum(1.0, a_max / a_norm)
                        ax, ay = ax * scale, ay * scale
                        return jnp.array([xt[2], xt[3], ax, ay])

                ilqr_agents.append(DI2DAgent(dt=dt, x_dim=4, u_dim=2,
                                            Q=jnp.eye(4)*1e-2, R=jnp.eye(2)*1e-2))
            else:
                raise ValueError(f"Unknown steering_type {stype}")
        
        hat_g_j = jnp.asarray(hat_g)                         # (N,2)
        w_goal_j = jnp.asarray(w.get("w_goal", 1.0))
        w_collision_j = jnp.asarray(w.get("w_collision", 10.0))
        w_control_j = jnp.asarray(w.get("w_control", 0.1))
        w_terminal_j = jnp.asarray(w.get("w_terminal", 1.0))
        r_safe_j = jnp.asarray(r_safe)
        
        def make_total_loss_i_cached(i):
            """Build per-agent total loss with outer-captured constants (single compile)."""
            def _softplus(x, alpha=20.0):
                return jnn.softplus(alpha * x) / alpha

            r2 = r_safe_j * r_safe_j
            def psi_sq(d2):
                return _softplus(r2 - d2)

            def total_loss_i(x_traj_i, u_traj_i, x_traj_all):
                # Precompute positions once: (N,T,2)
                pos_all = jnp.stack([xt[:, :2] for xt in x_traj_all], axis=0)
                pos_i = x_traj_i[:, :2]
                Nloc = pos_all.shape[0]
                mask = jnp.ones((Nloc,), dtype=pos_all.dtype).at[i].set(0.0)

                def body(acc, t):
                    # goal tracking
                    gt = w_goal_j * jnp.sum((pos_i[t] - hat_g_j[i]) ** 2)
                    # collision
                    diffs = pos_all[:, t, :] - pos_i[t]
                    d2 = jnp.sum(diffs * diffs, axis=1)
                    coll = w_collision_j * jnp.sum(mask * psi_sq(d2))
                    # control effort
                    ce = w_control_j * jnp.sum(u_traj_i[t] ** 2)
                    return acc + gt + coll + ce, None

                val, _ = lax.scan(body, 0.0, jnp.arange(x_traj_i.shape[0]))
                # terminal (shared evenly)
                x_T_all = [x_traj_all[k][-1] for k in range(Nloc)]
                val = val + (1.0 / Nloc) * w_terminal_j * terminal_loss(x_T_all)
                return val

            return total_loss_i

        # Compile and store gradient functions once
        dldx_list, dldu_list = [], []
        for i in range(N):
            L_i = make_total_loss_i_cached(i)
            dldx_list.append(jit(grad(L_i, argnums=0)))
            dldu_list.append(jit(grad(L_i, argnums=1)))
        
        # --- 7) Final assembly into
        # package model
        self.game_model = dict(
            N=N, T=T, dt=dt,
            dyn_list=dyn_list,
            x_dim_list=x_dim_list,
            u_dim_list=u_dim_list,
            x0_list=x0_list,
            u_traj_list=u_traj_list,
            runtime_losses=runtime_losses,
            terminal_loss=terminal_loss,
            hat_g=hat_g,
            g=g,
            weights=w,
            radius=r_safe,
            rollout_agent=rollout_agent,
            ilqr_agents=ilqr_agents,
            dldx_list=dldx_list,
            dldu_list=dldu_list,
            assignment_model=p.get("assignment_model", None),
        )
        
    def refresh_for_mpc(self, u_init_list=None, recompute_assignment=True):
        """
        Lightweight refresh for MPC:
        - Update x0_list from current self.params['robots'] states.
        - Optionally update u_traj_list with a warm-start (same shape as before).
        - Optionally recompute hat_g using the same rule as in construct_game.
        Assumes all static shapes/params (N, T, dims, dynamics) remain unchanged.
        """
        if self.game_model is None:
            raise RuntimeError("Call construct_game() once before refresh_for_mpc().")

        gm = self.game_model
        robots = self.params.get("robots", None)
        if robots is None:
            raise ValueError("params['robots'] must be set before refresh_for_mpc().")

        N, T = gm["N"], gm["T"]
        x_dim_list = gm["x_dim_list"]
        u_dim_list = gm["u_dim_list"]
        g = gm["g"]

        # 1) Rebuild x0_list from current robots (minimal state per steering type)
        def _x0_from_robot(rb: "Robot"):
            st = rb.state
            if rb.steering_type in ("bicycle", "unicycle"):
                return jnp.array([st["x"], st["y"], st["theta"]], dtype=float)
            elif rb.steering_type == "double-integrator":
                return jnp.array([st["x"], st["y"], st["v_x"], st["v_y"]], dtype=float)
            else:
                raise ValueError(f"Unknown steering_type {rb.steering_type}")

        x0_list = [ _x0_from_robot(rb) for rb in robots ]
        gm["x0_list"] = x0_list  # in-place refresh

        # 2) Warm start controls if provided (must keep shapes (T,u_dim) unchanged)
        if u_init_list is not None:
            if len(u_init_list) != N:
                raise ValueError(f"u_init_list length {len(u_init_list)} != N={N}")
            casted = []
            for k in range(N):
                U = jnp.asarray(u_init_list[k])
                if U.shape != (T, u_dim_list[k]):
                    raise ValueError(f"u_init_list[{k}] has shape {U.shape}, expected {(T, u_dim_list[k])}")
                casted.append(U)
            gm["u_traj_list"] = casted

        # 3) Optionally recompute assignment hat_g (priority: assignment_model > greedy NN)
        if recompute_assignment:
            if self.params.get("assignment_model", None) is not None:
                hat_g = self.params["assignment_model"].hat_g(g)
            else:
                x0_xy = jnp.stack([x0[:2] for x0 in x0_list], axis=0)  # (N,2)
                used = set()
                hat = []
                for i in range(N):
                    dists = jnp.linalg.norm(g - x0_xy[i], axis=1)
                    order = np.argsort(np.array(dists))
                    chosen = None
                    for idx in order:
                        if int(idx) not in used:
                            chosen = int(idx); break
                    if chosen is None: chosen = int(order[0])
                    used.add(chosen)
                    hat.append(g[chosen])
                hat_g = jnp.stack(hat, axis=0)
            gm["hat_g"] = hat_g

    def solve_game(self, warmup_only: bool = False, do_warmup: bool = True):
        """
        Solve Γ(P) via iLQGames-style iterations (requires lqrax).
        If 'lqrax' is unavailable, raises a clear error.

        Output stored in self.solution:
            - x_traj_list: list of (T, x_dim) arrays
            - u_traj_list: list of (T, u_dim) arrays
            - hat_g: (N,2) assigned goals actually tracked
            - P: if assignment_model exposes P(), include it; else None
            - params: echo of self.params for reproducibility
        """

        if self.game_model is None:
            raise RuntimeError("Please call construct_game() first.")

        gm = self.game_model
        N, T, dt = gm["N"], gm["T"], gm["dt"]
        dyn_list = gm["dyn_list"]
        x_dim_list = gm["x_dim_list"]
        u_dim_list = gm["u_dim_list"]
        x0_list = gm["x0_list"]
        u_traj_list = [jnp.array(U) for U in gm["u_traj_list"]]
        runtime_losses = gm["runtime_losses"]
        terminal_loss = gm["terminal_loss"]
        rollout_agent = gm["rollout_agent"]
        hat_g = gm["hat_g"]
        
        # --- Pull weights and constants into this scope (JAX-friendly scalars)
        w_dict = gm["weights"]
        w_goal = jnp.asarray(w_dict.get("w_goal", 1.0))
        w_collision = jnp.asarray(w_dict.get("w_collision", 10.0))
        w_control = jnp.asarray(w_dict.get("w_control", 0.1))
        w_terminal = jnp.asarray(w_dict.get("w_terminal", 1.0))

        r_safe = jnp.asarray(gm["radius"])
        hat_g = jnp.asarray(hat_g)  # ensure device array
        
        # print("[DEBUG] Initial x0_list:")
        # for i, x0 in enumerate(x0_list):
            # print(f"  Robot {i}:", x0,
            #     "  has_nan=", bool(jnp.isnan(x0).any()),
            #     "  has_inf=", bool(jnp.isinf(x0).any()))

        # print("[DEBUG] Initial u_traj_list stats (per agent):")
        # for i, U in enumerate(u_traj_list):
            # print(f"  Agent {i}: shape={tuple(U.shape)}, "
            #     f"mean={float(jnp.nanmean(U)):.6f}, "
            #     f"std={float(jnp.nanstd(U)):.6f}, "
            #     f"has_nan={bool(jnp.isnan(U).any())}, "
            #     f"has_inf={bool(jnp.isinf(U).any())}")

        # --- helpers ---
        def _block_tree(x):
            # Force JAX to finish async work for accurate timing
            return jtu.tree_map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x)
        
        def rollout_all(x0L, uL):
            return [rollout_agent(dyn_list[i], x0L[i], uL[i]) for i in range(N)]

        # initial rollout
        x_traj_list = rollout_all(x0_list, u_traj_list)
        
        # === PRE-GRAD DIAGNOSTIC: check rollout & raw loss ===
        for i, xt in enumerate(x_traj_list):
            if jnp.isnan(xt).any() or jnp.isinf(xt).any():
                bad = jnp.argwhere(jnp.isnan(xt) | jnp.isinf(xt))
                print(f"[NaN/Inf] rollout x_traj_list[{i}] bad indices (t, dim):", bad)

        # Check runtime loss without grad
        _any_nan_loss = False
        for i in range(N):
            rt = runtime_losses[i]
            for t in range(T):
                x_all_t = [x_traj_list[k][t] for k in range(N)]
                raw = rt(x_traj_list[i][t], u_traj_list[i][t], x_all_t)
                if not jnp.isfinite(raw):
                    _any_nan_loss = True
                    print(f"[NaN/Inf] runtime loss (pre-grad) at agent {i}, t={t}")
                    # 打印关键中间量
                    x_i = x_traj_list[i][t]
                    u_i = u_traj_list[i][t]
                    x_all_arr = jnp.stack(x_all_t, axis=0)
                    dists = jnp.linalg.norm(x_all_arr[:, :2] - x_i[:2], axis=1)
                    print("   x_i[:2]=", x_i[:2], "  u_i=", u_i)
                    print("   dists[min,max]=", float(jnp.nanmin(dists)), float(jnp.nanmax(dists)))
                    print("   any_nan dists=", bool(jnp.isnan(dists).any()), " any_inf dists=", bool(jnp.isinf(dists).any()))
                    # 可以顺带逐项打印 goal/coll/ctrl
                    # （与 make_runtime_loss 内的计算保持一致）
                    break
            if _any_nan_loss:
                break

        # Terminal loss check
        x_T_all = [x_traj_list[k][-1] for k in range(N)]
        term_val = terminal_loss(x_T_all)
        # print("[DEBUG] terminal_loss finite? ", bool(jnp.isfinite(term_val)))

        # per-agent horizon loss for gradients (sum of runtime + terminal)
        
        def make_total_loss_i(i: int):
            """Total horizon cost for agent i. Uses outer-scope JAX scalars and arrays."""
            # smooth hinge on squared distance to avoid sqrt cost (optional)
            def _softplus(x, alpha=20.0):
                return jnn.softplus(alpha * x) / alpha

            r2 = r_safe * r_safe
            def psi_sq(d2):
                # smooth max(0, r^2 - d^2)
                return _softplus(r2 - d2)

            def total_loss_i(x_traj_i, u_traj_i, x_traj_all):
                # Precompute (N, T, 2) positions once
                pos_all = jnp.stack([xt[:, :2] for xt in x_traj_all], axis=0)  # (N, T, 2)
                pos_i = x_traj_i[:, :2]                                       # (T, 2)

                # mask to remove self-collision
                mask = jnp.ones((N,), dtype=pos_all.dtype).at[i].set(0.0)

                def body(acc, t):
                    # goal tracking
                    gt = w_goal * jnp.sum((pos_i[t] - hat_g[i]) ** 2)

                    # collision (vectorized)
                    diffs = pos_all[:, t, :] - pos_i[t]        # (N, 2)
                    d2 = jnp.sum(diffs * diffs, axis=1)        # (N,)
                    coll = w_collision * jnp.sum(mask * psi_sq(d2))

                    # control effort
                    ce = w_control * jnp.sum(u_traj_i[t] ** 2)
                    return acc + gt + coll + ce, None

                val, _ = lax.scan(body, 0.0, jnp.arange(x_traj_i.shape[0]))

                # terminal loss shared evenly
                x_T_all = [x_traj_all[k][-1] for k in range(N)]
                val = val + (1.0 / N) * (w_terminal * jnp.sum(
                    jnp.stack([x_T_all[k][:2] for k in range(N)], axis=0) - hat_g
                )**2).sum() * 0.0 + (1.0 / N) * w_terminal * jnp.sum(
                    jnp.stack([x_T_all[k][:2] for k in range(N)], axis=0) - hat_g
                )**2  # keep structure similar; you也可直接调用原 terminal_loss

                return val

            return total_loss_i
        
        # JAX grads for linearized loss
        dldx_list = gm["dldx_list"]
        dldu_list = gm["dldu_list"]

        # --- optional warmup to JIT-compile key kernels BEFORE iterations ---
        if do_warmup:
            try:
                # 1) one linearize to get shapes
                A_w, B_w = [], []
                for i in range(N):
                    agent = gm["ilqr_agents"][i]
                    _, A_traj, B_traj = agent.linearize_dyn(x0_list[i], u_traj_list[i])
                    A_w.append(A_traj); B_w.append(B_traj)

                # 2) one gradient eval to compile dldx/dldu
                a_w, b_w = [], []
                for i in range(N):
                    a_w.append(dldx_list[i](x_traj_list[i], u_traj_list[i], x_traj_list))
                    b_w.append(dldu_list[i](x_traj_list[i], u_traj_list[i], x_traj_list))

                # 3) one LQ solve per agent
                for i in range(N):
                    agent = gm["ilqr_agents"][i]
                    _vu, _ = agent.solve(A_w[i], B_w[i], a_w[i], b_w[i])

                # 4) rollout and (optional) monitor once to compile those too
                _ = [rollout_agent(dyn_list[i], x0_list[i], u_traj_list[i]) for i in range(N)]
                if 'monitor_fn' in gm:
                    _ = gm['monitor_fn'](x_traj_list, u_traj_list)

                _block_tree((A_w, B_w, a_w, b_w))  # ensure async work finished
                print("[warmup] kernels compiled.")
            except Exception as _e:
                print("[WARN] warmup failed (ok to ignore):", _e)

        # return early if we only wanted to warm up
        if warmup_only:
            return


        # --- iLQGames outer loop ---
        num_iters = int(self.params.get("num_iters", 100))
        step_size = float(self.params.get("step_size", 1e-2))

        for it in range(num_iters):
            t_iter0 = _now()

            # 1) linearize dynamics
            t0 = _now()
            A_list, B_list = [], []
            for i in range(N):
                agent = gm["ilqr_agents"][i]
                x_traj_i, A_traj, B_traj = agent.linearize_dyn(x0_list[i], u_traj_list[i])
                # Assign and collect
                x_traj_list[i] = x_traj_i
                A_list.append(A_traj); B_list.append(B_traj)
            _block_tree((x_traj_list, A_list, B_list))  # ensure timing correctness
            t_lin = _now() - t0

            # 2) linearize losses (gradients)
            t0 = _now()
            a_list, b_list = [], []
            for i in range(N):
                a_traj = dldx_list[i](x_traj_list[i], u_traj_list[i], x_traj_list)
                b_traj = dldu_list[i](x_traj_list[i], u_traj_list[i], x_traj_list)
                a_list.append(a_traj); b_list.append(b_traj)
            _block_tree((a_list, b_list))
            t_grad = _now() - t0

            # 3) solve LQ subproblem
            t0 = _now()
            v_u_list = []
            for i in range(N):
                agent = gm["ilqr_agents"][i]
                v_u, _ = agent.solve(A_list[i], B_list[i], a_list[i], b_list[i])
                v_u_list.append(v_u)
            _block_tree(v_u_list)
            t_solve = _now() - t0

            # 4) update controls
            t0 = _now()
            for i in range(N):
                u_traj_list[i] = u_traj_list[i] + step_size * v_u_list[i]
            # Pure numpy/JAX host ops; no need to block but harmless:
            _block_tree(u_traj_list)
            t_update = _now() - t0

            # 5) rollout new trajectories
            t0 = _now()
            x_traj_list = [rollout_agent(dyn_list[i], x0_list[i], u_traj_list[i]) for i in range(N)]
            _block_tree(x_traj_list)
            t_roll = _now() - t0

            # (optional) monitoring / early stop
            t0 = _now()
            # if it % 10 == 0:
            #     pass
            #     tot = 0.0
            #     for i in range(N):
            #         rt = runtime_losses[i]
            #         for t in range(T):
            #             x_all_t = [x_traj_list[k][t] for k in range(N)]
            #             tot_it = rt(x_traj_list[i][t], u_traj_list[i][t], x_all_t)
            #             tot = tot + tot_it
            #     tot = tot + terminal_loss([x_traj_list[i][-1] for i in range(N)])
            #     _block_tree(tot)
            t_monitor = _now() - t0

            t_iter = _now() - t_iter0

            # Pretty timing print (ms)
            # print(
            #     f"[TIMING] it={it+1:03d} | "
            #     f"linearize={t_lin*1e3:7.2f} ms | "
            #     f"grads={t_grad*1e3:7.2f} ms | "
            #     f"solve={t_solve*1e3:7.2f} ms | "
            #     f"update={t_update*1e3:7.2f} ms | "
            #     f"rollout={t_roll*1e3:7.2f} ms | "
            #     f"monitor={t_monitor*1e3:7.2f} ms | "
            #     f"total={t_iter*1e3:7.2f} ms"
            # )

        self.solution = dict(
            x_traj_list=x_traj_list,
            u_traj_list=u_traj_list,
            hat_g=hat_g,
            P=(self.game_model["assignment_model"].P() 
            if self.game_model.get("assignment_model", None) is not None 
            and hasattr(self.game_model["assignment_model"], "P") else None),
            params=self.params,
        )
