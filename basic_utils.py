# basic_utils.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
from PIL import Image


def visualize_scene(robots, env, figsize=(8, 6), filename=None):
    """
    Visualize robots (filled circles with index labels), velocity/heading,
    environment bounds, and goal positions. Legend for robots and goals is
    placed outside the plot area on the right.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize)

    # --- Draw environment bounds ---
    A = np.array(env.bounds.get("A", []))
    b = np.array(env.bounds.get("b", []))
    if len(A) == 4 and len(b) == 4:
        # rectangle encoding: [x<=xmax, -x<=-xmin, y<=ymax, -y<=-ymin]
        xmax, xmin, ymax, ymin = b[0], -b[1], b[2], -b[3]
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=1.5, edgecolor="black", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)

    # --- Draw goals (and keep handle for legend) ---
    goals_handle = None
    if getattr(env, "goals", None):
        goals = np.array(env.goals)
        sc = ax.scatter(goals[:, 0], goals[:, 1], s=60, c="red", marker="x", label="goals", zorder=5)
        goals_handle = Line2D([0], [0], color="red", marker="x", linestyle="None", markersize=8, label="goals")

    # --- Draw robots (and keep a single proxy handle for legend) ---
    robot_handle = patches.Circle((0, 0), radius=0.3, color="tab:blue", alpha=0.6, label="robots")
    for r in robots:
        x = r.state.get("x", 0.0)
        y = r.state.get("y", 0.0)
        theta = r.state.get("theta", 0.0)
        # prefer planar speed; fall back to v_x magnitude if needed
        v = r.state.get("v", None)
        if v is None:
            vx = r.state.get("v_x", 0.0)
            vy = r.state.get("v_y", 0.0)
            v = float(np.hypot(vx, vy))

        radius = float(r.params.get("radius", 0.3))
        circle = plt.Circle((x, y), radius, color="tab:blue", alpha=0.6, zorder=3)
        ax.add_patch(circle)

        # index label
        ax.text(x, y, f"{r.index}", color="white", ha="center", va="center",
                fontsize=9, weight="bold", zorder=6)

        # velocity/heading arrow (direction scaled by speed)
        dx = np.cos(theta) * v * 0.5
        dy = np.sin(theta) * v * 0.5
        ax.arrow(
            x, y, dx, dy,
            head_width=0.15, head_length=0.25,
            fc="k", ec="k", alpha=0.8, length_includes_head=True, zorder=4,
        )

    # --- Formatting ---
    ax.set_aspect("equal")
    ax.set_title(f"Formation: {env.formation}, Robots: {len(robots)}", fontsize=12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Build legend outside the axes (right side)
    handles = []
    if robot_handle is not None:
        handles.append(robot_handle)
    if goals_handle is not None:
        handles.append(goals_handle)

    # Make room on the right for the legend
    plt.subplots_adjust(right=0.80)
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.02),  # move to outside top-right corner
        frameon=False,
    )

    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)
    plt.close(fig)


def _render_scene_frame(robots, env, figsize=(8, 6), goals_override=None, formation_label=None):
    """
    Render a single scene frame to a PIL Image.
    If goals_override is provided (array-like Nx2), use it instead of env.goals.
    If formation_label is provided, use it in title instead of env.formation.
    """
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=figsize)

    # --- Draw environment bounds ---
    A = np.array(env.bounds.get("A", []))
    b = np.array(env.bounds.get("b", []))
    if len(A) == 4 and len(b) == 4:
        xmax, xmin, ymax, ymin = b[0], -b[1], b[2], -b[3]
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=1.5, edgecolor="black", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)

    # --- Draw goals (from override if given) ---
    goals_handle = None
    goals_to_draw = None
    if goals_override is not None:
        goals_to_draw = np.asarray(goals_override, dtype=float)
    elif getattr(env, "goals", None):
        goals_to_draw = np.asarray(env.goals, dtype=float)

    if goals_to_draw is not None and len(goals_to_draw) > 0:
        ax.scatter(goals_to_draw[:, 0], goals_to_draw[:, 1], s=60, c="red", marker="x", label="goals", zorder=5)
        from matplotlib.lines import Line2D
        goals_handle = Line2D([0], [0], color="red", marker="x", linestyle="None", markersize=8, label="goals")

    # --- Draw robots ---
    robot_handle = patches.Circle((0, 0), radius=0.3, color="tab:blue", alpha=0.6, label="robots")
    for r in robots:
        x = r.state.get("x", 0.0)
        y = r.state.get("y", 0.0)
        theta = r.state.get("theta", 0.0)
        v = r.state.get("v", None)
        if v is None:
            vx = r.state.get("v_x", 0.0)
            vy = r.state.get("v_y", 0.0)
            v = float(np.hypot(vx, vy))

        radius = float(r.params.get("radius", 0.3))
        circle = plt.Circle((x, y), radius, color="tab:blue", alpha=0.6, zorder=3)
        ax.add_patch(circle)

        ax.text(x, y, f"{r.index}", color="white", ha="center", va="center",
                fontsize=9, weight="bold", zorder=6)

        dx = np.cos(theta) * v * 0.5
        dy = np.sin(theta) * v * 0.5
        ax.arrow(
            x, y, dx, dy,
            head_width=0.15, head_length=0.25,
            fc="k", ec="k", alpha=0.8, length_includes_head=True, zorder=4,
        )

    # --- Formatting ---
    ax.set_aspect("equal")
    title_str = f"Formation: {formation_label if formation_label is not None else getattr(env, 'formation', '')}, Robots: {len(robots)}"
    ax.set_title(title_str, fontsize=12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle=":", alpha=0.5)

    handles = []
    if robot_handle is not None:
        handles.append(robot_handle)
    if goals_handle is not None:
        handles.append(goals_handle)

    plt.subplots_adjust(right=0.80)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.02), frameon=False)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def visualize_scene_animation(
    robot_snapshots, env, filename="animation.gif", figsize=(8, 6),
    duration=0.1, step_label=True,
    goals_snapshots=None, formation_labels=None
):
    """
    Create an animated GIF from snapshots.
    If goals_snapshots is provided, it must be the same length as robot_snapshots.
    Each entry provides the goals for that frame.
    formation_labels, if provided, supplies the title label per frame.
    """
    if not robot_snapshots:
        raise ValueError("robot_snapshots list cannot be empty")

    use_goals = goals_snapshots is not None
    use_form = formation_labels is not None
    if use_goals and len(goals_snapshots) != len(robot_snapshots):
        raise ValueError("goals_snapshots length must match robot_snapshots length")
    if use_form and len(formation_labels) != len(robot_snapshots):
        raise ValueError("formation_labels length must match robot_snapshots length")

    frames = []
    for step, robots in enumerate(robot_snapshots):
        goals_override = goals_snapshots[step] if use_goals else None
        formation_label = formation_labels[step] if use_form else None

        img = _render_scene_frame(
            robots, env, figsize=figsize,
            goals_override=goals_override,
            formation_label=formation_label
        )

        if step_label:
            from PIL import ImageDraw, ImageFont
            import matplotlib.font_manager as fm
            draw = ImageDraw.Draw(img)
            step_text = f"Step: {step}"
            text_x, text_y = 10, 10
            try:
                font_path = fm.findfont("DejaVu Sans", fallback_to_default=True)
                font = ImageFont.truetype(font_path, 16)
            except Exception:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
                except Exception:
                    font = ImageFont.load_default()

            try:
                bbox = draw.textbbox((text_x, text_y), step_text, font=font)
            except Exception:
                w, h = draw.textsize(step_text, font=font)
                bbox = (text_x, text_y, text_x + w, text_y + h)

            padding = 5
            draw.rectangle([bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding], fill=(255, 255, 255))
            draw.text((text_x, text_y), step_text, fill=(0, 0, 0), font=font)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        frames.append(img)

    if len(frames) > 0:
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=int(duration * 1000),
            loop=0
        )
        print(f"[info] Created animation with {len(frames)} frames: {filename}")


def _bounds_rect_from_Ab(bounds):
    """
    Extract an axis-aligned rectangle (xmin, ymin, xmax, ymax) from A x <= b
    assuming the rectangle encoding:
      rows: [x<=xmax, -x<=-xmin, y<=ymax, -y<=-ymin]
    """
    A = bounds.get("A", [])
    b = bounds.get("b", [])
    if len(A) == 4 and len(b) == 4:
        xmax = float(b[0])
        xmin = -float(b[1])
        ymax = float(b[2])
        ymin = -float(b[3])
        return xmin, ymin, xmax, ymax
    # Fallback window
    return -5.0, -3.0, 5.0, 3.0

def _sample_nonoverlapping_poses(
    n: int,
    rect: tuple[float, float, float, float],
    radii: list[float],
    rng: np.random.Generator,
    max_trials_per_robot: int = 5000,
    angle_uniform: bool = True,
    margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rejection-sample n poses (x, y, theta) inside 'rect' such that
    pairwise center distances satisfy d_ij >= r_i + r_j + margin.

    - rect = (xmin, ymin, xmax, ymax)
    - radii: list of robot radii (length n)
    - margin: extra slack to avoid borderline overlaps
    - theta is sampled U(-pi, pi) if angle_uniform is True, else zeros
    """
    xmin, ymin, xmax, ymax = rect
    xs = np.empty(n, dtype=float)
    ys = np.empty(n, dtype=float)
    thetas = np.empty(n, dtype=float)

    # shrink the sampling box by each robot's radius to keep circles inside
    for i in range(n):
        r_i = radii[i]
        # fallback to a small positive if a radius is missing/zero
        r_i = float(r_i) if r_i > 0.0 else 1e-6

        # try to place robot i
        placed = False
        for _ in range(max_trials_per_robot):
            x = rng.uniform(xmin + r_i, xmax - r_i)
            y = rng.uniform(ymin + r_i, ymax - r_i)

            ok = True
            for j in range(i):
                dx = x - xs[j]
                dy = y - ys[j]
                dij2 = dx * dx + dy * dy
                req = (r_i + radii[j] + margin)
                if dij2 < req * req:
                    ok = False
                    break
            if ok:
                xs[i] = x
                ys[i] = y
                thetas[i] = rng.uniform(-np.pi, np.pi) if angle_uniform else 0.0
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Failed to place robot {i} without overlap. "
                "Consider fewer robots, smaller radii, or a larger rectangle."
            )

    return xs, ys, thetas

