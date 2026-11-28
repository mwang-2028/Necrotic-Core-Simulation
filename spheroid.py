import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import EllipseCollection
from scipy.spatial import cKDTree


R0 = 5.0
R_MAX = np.sqrt(2) * R0
K_GROW = (R_MAX - R0) / 18.0
GROW_SPREAD = 0.25
DAUGHTER_DISTANCE = 1.5

K_COLLISION = 5000.0
FRICTION = 20.0
FRICTION_POWER = 1
MASS = 1.0

BOX = 1200.0
GRID_SPACING = 5.0
GRID_N = int(round(BOX / GRID_SPACING)) + 1
NUTRIENT_START = 1.0
CONSUMPTION = 0.02
STARVE_THRESHOLD = 1.0
T_DEAD = 12.0
T_DEAD_SD = 3.0

DT = 0.01
TARGET_CELLS = 5000
MAX_HOURS = 1000


def make_first_cell(rng):
    return {
        "x": np.zeros(1),
        "y": np.zeros(1),
        "vx": np.zeros(1),
        "vy": np.zeros(1),
        "r": np.array([R0]),
        "k": K_GROW * (1 + GROW_SPREAD * rng.uniform(-1, 1, 1)),
        "alive": np.array([True]),
        "starve": np.zeros(1),
        "t_dead": survival_time(rng, 1),
    }


def survival_time(rng, n):
    return np.maximum(rng.normal(T_DEAD, T_DEAD_SD, n), DT)


def make_nutrients():
    return np.full((GRID_N, GRID_N), NUTRIENT_START)


def grow(cells):
    alive = cells["alive"]
    cells["r"][alive] = np.minimum(cells["r"][alive] + cells["k"][alive] * DT, R_MAX)


def divide(cells, rng):
    ready = np.nonzero(cells["alive"] & (cells["r"] >= R_MAX))[0]
    if len(ready) == 0:
        return ready, ready
    r_daughter = cells["r"][ready] / np.sqrt(2)
    d = DAUGHTER_DISTANCE * r_daughter
    angle = rng.uniform(0, 2 * np.pi, len(ready))

    new = {
        "x": cells["x"][ready] + d * np.cos(angle),
        "y": cells["y"][ready] + d * np.sin(angle),
        "vx": cells["vx"][ready].copy(),
        "vy": cells["vy"][ready].copy(),
        "r": r_daughter,
        "k": K_GROW * (1 + GROW_SPREAD * rng.uniform(-1, 1, len(ready))),
        "alive": np.ones(len(ready), dtype=bool),
        "starve": cells["starve"][ready].copy(),
        "t_dead": survival_time(rng, len(ready)),
    }
    cells["r"][ready] = r_daughter
    cells["k"][ready] = K_GROW * (1 + GROW_SPREAD * rng.uniform(-1, 1, len(ready)))
    first_new = len(cells["x"])
    for key in cells:
        cells[key] = np.concatenate([cells[key], new[key]])
    return ready, np.arange(first_new, first_new + len(ready))


def collision_forces(cells):
    n = len(cells["x"])
    fx, fy = np.zeros(n), np.zeros(n)
    if n < 2:
        return fx, fy

    points = np.column_stack([cells["x"], cells["y"]])
    pairs = cKDTree(points).query_pairs(2 * R_MAX, output_type="ndarray")
    if len(pairs) == 0:
        return fx, fy
    i, j = pairs[:, 0], pairs[:, 1]

    dx = cells["x"][i] - cells["x"][j]
    dy = cells["y"][i] - cells["y"][j]
    d = np.sqrt(dx**2 + dy**2)
    touching = d < cells["r"][i] + cells["r"][j]
    i, j, dx, dy, d = i[touching], j[touching], dx[touching], dy[touching], d[touching]

    d = np.maximum(d, 0.1 * R0)
    f = K_COLLISION / d
    pair_fx = f * dx / d
    pair_fy = f * dy / d

    np.add.at(fx, i, pair_fx)
    np.add.at(fy, i, pair_fy)
    np.add.at(fx, j, -pair_fx)
    np.add.at(fy, j, -pair_fy)
    return fx, fy


def friction_strength(cells):
    speed = np.sqrt(cells["vx"]**2 + cells["vy"]**2)
    return FRICTION * speed**(FRICTION_POWER - 1)


def move(cells):
    fx, fy = collision_forces(cells)
    c = friction_strength(cells)
    slow_down = 1 + c * DT / MASS
    cells["vx"] = (cells["vx"] + fx / MASS * DT) / slow_down
    cells["vy"] = (cells["vy"] + fy / MASS * DT) / slow_down
    cells["x"] += cells["vx"] * DT
    cells["y"] += cells["vy"] * DT


def covered_points(cells, ids):
    x, y, r = cells["x"][ids], cells["y"][ids], cells["r"][ids]
    reach = int(np.ceil(R_MAX / GRID_SPACING + 0.5))
    steps = np.arange(-reach, reach + 1)
    ox, oy = np.meshgrid(steps, steps)
    ox, oy = ox.ravel(), oy.ravel()

    gi = np.rint((x + BOX / 2) / GRID_SPACING).astype(int)[:, None] + ox
    gj = np.rint((y + BOX / 2) / GRID_SPACING).astype(int)[:, None] + oy
    px = gi * GRID_SPACING - BOX / 2
    py = gj * GRID_SPACING - BOX / 2
    inside = (px - x[:, None])**2 + (py - y[:, None])**2 <= r[:, None]**2
    inside &= (gi >= 0) & (gi < GRID_N) & (gj >= 0) & (gj < GRID_N)

    owner, spot = np.nonzero(inside)
    return owner, gi[owner, spot] * GRID_N + gj[owner, spot]


def feed(cells, nutrients):
    alive = np.nonzero(cells["alive"])[0]
    if len(alive) == 0:
        return
    owner, point = covered_points(cells, alive)
    u = nutrients.reshape(-1)
    eaters = np.bincount(point, minlength=u.size)
    u -= CONSUMPTION * DT * eaters
    np.maximum(u, 0, out=u)

    food = np.bincount(owner, weights=u[point], minlength=len(alive))
    starve = np.where(food < STARVE_THRESHOLD, cells["starve"][alive] + DT, 0.0)
    cells["starve"][alive] = starve
    cells["alive"][alive[starve >= cells["t_dead"][alive]]] = False


def measure(cells, hour):
    x, y, r, alive = cells["x"], cells["y"], cells["r"], cells["alive"]
    cx, cy = x.mean(), y.mean()
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    dead = ~alive
    radius = float((distance + r).max())
    necrotic_radius = float(np.percentile(distance[dead], 95) + R0) if dead.any() else 0.0
    return {
        "hour": hour,
        "cells": len(x),
        "alive": int(alive.sum()),
        "dead": int(dead.sum()),
        "radius": radius,
        "necrotic_radius": necrotic_radius,
        "rim_width": radius - necrotic_radius,
        "average_speed": float(np.sqrt(cells["vx"]**2 + cells["vy"]**2).mean()),
    }


def run_simulation(target_cells=TARGET_CELLS, max_hours=MAX_HOURS, seed=1,
                   snapshot_counts=(), frame_every=None, verbose=False):
    rng = np.random.default_rng(seed)
    cells = make_first_cell(rng)
    nutrients = make_nutrients()
    history, snapshots, frames = [], {}, []
    waiting = sorted(snapshot_counts)
    steps_per_hour = round(1 / DT)

    step = 0
    while True:
        hour = step * DT
        if step % steps_per_hour == 0:
            stats = measure(cells, hour)
            history.append(stats)
            if verbose and step % (steps_per_hour * 24) == 0:
                print(f"  hour {hour:5.0f}: {stats['cells']:5d} cells "
                      f"({stats['dead']} dead), spheroid radius {stats['radius']:5.0f} um")
            if frame_every and step % (steps_per_hour * frame_every) == 0:
                frames.append((hour, {k: v.copy() for k, v in cells.items()}, nutrients.copy()))
        while waiting and len(cells["x"]) >= waiting[0]:
            snapshots[waiting.pop(0)] = (hour, {k: v.copy() for k, v in cells.items()},
                                         nutrients.copy())
        too_big = history[-1]["radius"] > 0.45 * BOX
        if len(cells["x"]) >= target_cells or hour >= max_hours or too_big:
            break

        grow(cells)
        divide(cells, rng)
        move(cells)
        feed(cells, nutrients)
        step += 1

    history.append(measure(cells, step * DT))
    if frame_every:
        frames.append((step * DT, {k: v.copy() for k, v in cells.items()}, nutrients.copy()))
    return {"history": history, "snapshots": snapshots, "frames": frames, "final": cells,
            "nutrients": nutrients}


def draw_cells(ax, cells, title, nutrients=None, reach=None):
    x, y, r, alive = cells["x"], cells["y"], cells["r"], cells["alive"]
    if reach is None:
        reach = max(np.abs(x).max(), np.abs(y).max()) + R_MAX * 2
    image = None
    if nutrients is not None:
        edge = BOX / 2 + GRID_SPACING / 2
        image = ax.imshow(nutrients.T, origin="lower", extent=(-edge, edge, -edge, edge),
                          cmap="Greens", vmin=0, vmax=NUTRIENT_START, alpha=0.6)
    colors = np.where(alive, "#e8a0a0", "#555555")
    edges = np.where(alive, "#a83232", "#222222")
    circles = EllipseCollection(2 * r, 2 * r, np.zeros_like(r), units="xy",
                                offsets=np.column_stack([x, y]), offset_transform=ax.transData,
                                facecolors=colors, edgecolors=edges, linewidths=0.4)
    ax.add_collection(circles)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.set_title(title)
    return image


if __name__ == "__main__":
    WATCH_COUNTS = (2, 8, 50, 500, 2000, TARGET_CELLS)
    FRAME_EVERY_HOURS = 3
    VIEW = 0.45 * BOX
    SHOW_WINDOWS = True
    print(f"Growing from 1 cell to {TARGET_CELLS} cells...")
    run = run_simulation(snapshot_counts=WATCH_COUNTS, frame_every=FRAME_EVERY_HOURS,
                         verbose=True)
    history = run["history"]
    end = history[-1]

    columns = ["hour", "cells", "alive", "dead", "radius", "necrotic_radius", "rim_width",
               "average_speed"]
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for h in history:
            writer.writerow([h[c] if isinstance(h[c], int) else round(float(h[c]), 3)
                             for c in columns])

    hours = [h["hour"] for h in history]
    has_core = [h["dead"] > 0 for h in history]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(hours, [h["cells"] for h in history], color="#333333", label="all")
    axes[0].plot(hours, [h["alive"] for h in history], color="#a83232", label="living")
    axes[0].plot(hours, [max(h["dead"], 0.5) for h in history], color="#777777",
                 linestyle="--", label="dead")
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=1)
    axes[0].set_title("Number of cells")
    axes[0].legend()
    axes[1].plot(hours, [h["radius"] for h in history], color="#2a5aa8", label="spheroid")
    axes[1].plot(hours, [h["necrotic_radius"] for h in history], color="#555555",
                 label="necrotic core")
    axes[1].set_title("Radius")
    axes[1].set_ylabel("um")
    axes[1].legend()
    axes[2].plot([t for t, c in zip(hours, has_core) if c],
                 [h["rim_width"] for h, c in zip(history, has_core) if c], color="#a83232")
    axes[2].set_title("Living rim width (after the core appears)")
    axes[2].set_ylabel("um")
    axes[3].plot(hours, [h["average_speed"] for h in history], color="#2a7a3a")
    axes[3].set_title("Average cell speed")
    axes[3].set_ylabel("um per hour")
    for ax in axes:
        ax.set_xlabel("time (hours)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("tracked_variables.png", dpi=120)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), layout="constrained")
    draw_cells(axes[0], run["final"], f"Hour {end['hour']:.0f}: {end['alive']} living, "
               f"{end['dead']} dead", run["nutrients"], reach=VIEW)
    edge = BOX / 2 + GRID_SPACING / 2
    image = axes[1].imshow(run["nutrients"].T, origin="lower", extent=(-edge, edge, -edge, edge),
                           cmap="Greens", vmin=0, vmax=NUTRIENT_START)
    axes[1].set_xlim(-VIEW, VIEW)
    axes[1].set_ylim(-VIEW, VIEW)
    axes[1].set_title("Nutrient left (cells not drawn)")
    fig.colorbar(image, ax=axes[1], label="nutrient (1 = starting amount)")
    for ax in axes:
        ax.set_xlabel("um")
        ax.set_ylabel("um")
    fig.savefig("snapshot.png", dpi=150)

    counts = [c for c in WATCH_COUNTS if c in run["snapshots"]]
    fig, axes = plt.subplots(1, len(counts), figsize=(3.2 * len(counts) + 1, 3.6),
                             layout="constrained")
    for ax, count in zip(axes, counts):
        hour, cells, nutrients = run["snapshots"][count]
        image = draw_cells(ax, cells, f"{len(cells['x'])} cells, hour {hour:.0f}", nutrients)
        ax.tick_params(labelsize=7)
    fig.colorbar(image, ax=axes, label="nutrient")
    fig.savefig("over_time.png", dpi=120)

    fig, ax = plt.subplots(figsize=(6, 6))

    def draw_frame(frame):
        ax.clear()
        hour, cells, nutrients = frame
        draw_cells(ax, cells, f"hour {hour:.0f}: {cells['alive'].sum()} living, "
                   f"{(~cells['alive']).sum()} dead", nutrients, reach=VIEW)

    growth = FuncAnimation(fig, draw_frame, frames=run["frames"])
    growth.save("growth.gif", writer=PillowWriter(fps=8))
    plt.close(fig)

    print(f"Hour {end['hour']:.0f}: {end['cells']} cells ({end['alive']} living, "
          f"{end['dead']} dead), spheroid radius {end['radius']:.0f} um, "
          f"necrotic core radius {end['necrotic_radius']:.0f} um, "
          f"living rim width {end['rim_width']:.0f} um")
    print("Saved results.csv, tracked_variables.png, snapshot.png, over_time.png and growth.gif")

    if SHOW_WINDOWS:
        print("Showing the plots. Close the windows to finish.")
        plt.show()
