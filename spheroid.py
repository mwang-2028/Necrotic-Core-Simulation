import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from scipy.spatial import cKDTree


R0 = 5.0
R_MAX = np.sqrt(2) * R0
K_GROW = (R_MAX - R0) / 18.0
GROW_SPREAD = 0.25

K_COLLISION = 20000.0
FRICTION = 20.0
FRICTION_POWER = 1
MASS = 1.0

DT = 0.02
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
    }


def grow(cells):
    cells["r"] = np.minimum(cells["r"] + cells["k"] * DT, R_MAX)


def divide(cells, rng):
    ready = np.nonzero(cells["r"] >= R_MAX)[0]
    if len(ready) == 0:
        return
    angle = rng.uniform(0, 2 * np.pi, len(ready))
    offset = R_MAX - R0
    dx, dy = offset * np.cos(angle), offset * np.sin(angle)

    new = {
        "x": cells["x"][ready] - dx,
        "y": cells["y"][ready] - dy,
        "vx": cells["vx"][ready].copy(),
        "vy": cells["vy"][ready].copy(),
        "r": np.full(len(ready), R0),
        "k": K_GROW * (1 + GROW_SPREAD * rng.uniform(-1, 1, len(ready))),
    }
    cells["x"][ready] += dx
    cells["y"][ready] += dy
    cells["r"][ready] = R0
    cells["k"][ready] = K_GROW * (1 + GROW_SPREAD * rng.uniform(-1, 1, len(ready)))
    for key in cells:
        cells[key] = np.concatenate([cells[key], new[key]])


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


def measure(cells, hour):
    x, y, r = cells["x"], cells["y"], cells["r"]
    cx, cy = x.mean(), y.mean()
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    return {
        "hour": hour,
        "cells": len(x),
        "radius": float((distance + r).max()),
        "average_cell_radius": float(r.mean()),
        "average_speed": float(np.sqrt(cells["vx"]**2 + cells["vy"]**2).mean()),
    }


def run_simulation(target_cells=TARGET_CELLS, max_hours=MAX_HOURS, seed=1,
                   snapshot_counts=(), verbose=False):
    rng = np.random.default_rng(seed)
    cells = make_first_cell(rng)
    history, snapshots = [], {}
    waiting = sorted(snapshot_counts)
    steps_per_hour = round(1 / DT)

    step = 0
    while True:
        hour = step * DT
        if step % steps_per_hour == 0:
            stats = measure(cells, hour)
            history.append(stats)
            if verbose and step % (steps_per_hour * 24) == 0:
                print(f"  hour {hour:5.0f}: {stats['cells']:5d} cells, "
                      f"spheroid radius {stats['radius']:5.0f} um")
        while waiting and len(cells["x"]) >= waiting[0]:
            snapshots[waiting.pop(0)] = (hour, {k: v.copy() for k, v in cells.items()})
        if len(cells["x"]) >= target_cells or hour >= max_hours:
            break

        grow(cells)
        divide(cells, rng)
        move(cells)
        step += 1

    history.append(measure(cells, step * DT))
    return {"history": history, "snapshots": snapshots, "final": cells}


def draw_cells(ax, cells, title):
    x, y, r = cells["x"], cells["y"], cells["r"]
    circles = EllipseCollection(2 * r, 2 * r, np.zeros_like(r), units="xy",
                                offsets=np.column_stack([x, y]), offset_transform=ax.transData,
                                facecolors="#e8a0a0", edgecolors="#a83232", linewidths=0.4)
    ax.add_collection(circles)
    reach = max(np.abs(x).max(), np.abs(y).max()) + R_MAX * 2
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.set_title(title)


if __name__ == "__main__":
    WATCH_COUNTS = (2, 8, 50, 500, 2000, TARGET_CELLS)
    SHOW_WINDOWS = True
    print(f"Growing from 1 cell to {TARGET_CELLS} cells...")
    run = run_simulation(snapshot_counts=WATCH_COUNTS, verbose=True)
    history = run["history"]
    end = history[-1]

    columns = ["hour", "cells", "radius", "average_cell_radius", "average_speed"]
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for h in history:
            writer.writerow([h[c] if isinstance(h[c], int) else round(float(h[c]), 3)
                             for c in columns])

    hours = [h["hour"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(hours, [h["cells"] for h in history], color="#a83232")
    axes[0].set_yscale("log")
    axes[0].set_title("Number of cells")
    axes[1].plot(hours, [h["radius"] for h in history], color="#2a5aa8")
    axes[1].set_title("Spheroid radius")
    axes[1].set_ylabel("um")
    axes[2].plot(hours, [h["average_speed"] for h in history], color="#2a7a3a")
    axes[2].set_title("Average cell speed")
    axes[2].set_ylabel("um per hour")
    for ax in axes:
        ax.set_xlabel("time (hours)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("tracked_variables.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_cells(ax, run["final"], f"Hour {end['hour']:.0f}: {end['cells']} cells")
    ax.set_xlabel("um")
    fig.tight_layout()
    fig.savefig("snapshot.png", dpi=150)

    counts = [c for c in WATCH_COUNTS if c in run["snapshots"]]
    fig, axes = plt.subplots(1, len(counts), figsize=(3.2 * len(counts), 3.6))
    for ax, count in zip(axes, counts):
        hour, cells = run["snapshots"][count]
        draw_cells(ax, cells, f"{len(cells['x'])} cells, hour {hour:.0f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("over_time.png", dpi=120)

    print(f"Hour {end['hour']:.0f}: {end['cells']} cells, spheroid radius {end['radius']:.0f} um")
    print("Saved results.csv, tracked_variables.png, snapshot.png and over_time.png")

    if SHOW_WINDOWS:
        print("Showing the plots. Close the windows to finish.")
        plt.show()
