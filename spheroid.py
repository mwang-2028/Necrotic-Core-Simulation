import csv
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


GRID_SIZE = 100
DX = 20.0
HOURS = 480

O2_MEDIUM = 100.0
O2_DIFFUSION = 2000.0
O2_USE = 15.0
O2_HALF = 1.0
O2_DIVIDE = 10.0
O2_DEATH = 1.0

CELL_CYCLE_HOURS = 24.0
HYPOXIC_DEATH_CHANCE = 0.2

EMPTY, ALIVE, NECROTIC = 0, 1, 2


def solve_oxygen_field(inside, uptake):
    field = np.full(inside.shape, O2_MEDIUM, dtype=float)
    rows, cols = np.nonzero(inside)
    n = len(rows)
    if n == 0:
        return field

    number = -np.ones(inside.shape, dtype=int)
    number[rows, cols] = np.arange(n)

    a = O2_DIFFUSION / DX**2
    diagonal = -4 * a - uptake[rows, cols]
    right_side = np.zeros(n)
    matrix_rows, matrix_cols, matrix_vals = [], [], []

    for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        n_rows, n_cols = rows + d_row, cols + d_col
        neighbor_inside = inside[n_rows, n_cols]
        matrix_rows.append(np.arange(n)[neighbor_inside])
        matrix_cols.append(number[n_rows, n_cols][neighbor_inside])
        matrix_vals.append(np.full(neighbor_inside.sum(), a))
        right_side = right_side - a * O2_MEDIUM * (~neighbor_inside)

    matrix_rows.append(np.arange(n))
    matrix_cols.append(np.arange(n))
    matrix_vals.append(diagonal)
    A = scipy.sparse.csr_matrix(
        (np.concatenate(matrix_vals), (np.concatenate(matrix_rows), np.concatenate(matrix_cols))),
        shape=(n, n),
    )
    field[rows, cols] = scipy.sparse.linalg.spsolve(A, right_side)
    return field


def solve_oxygen(cells, oxygen_guess):
    inside = cells != EMPTY
    oxygen = oxygen_guess
    for _ in range(4):
        uptake = np.where(cells == ALIVE, O2_USE / (np.maximum(oxygen, 0) + O2_HALF), 0.0)
        oxygen = solve_oxygen_field(inside, uptake)
    return np.maximum(oxygen, 0)


def random_empty_neighbor(cells, row, col, rng):
    options = []
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            r, c = row + d_row, col + d_col
            if cells[r, c] == EMPTY:
                options.append((r, c))
    if not options:
        return None
    return options[rng.integers(len(options))]


def measure(cells, oxygen):
    area_per_square = DX**2
    inside = cells != EMPTY
    radius = np.sqrt(np.sum(inside) * area_per_square / np.pi)
    core_radius = np.sqrt(np.sum(cells == NECROTIC) * area_per_square / np.pi)
    return {
        "radius": radius,
        "core_radius": core_radius,
        "rim_thickness": radius - core_radius,
        "average_oxygen": oxygen[inside].mean(),
        "lowest_oxygen": oxygen[inside].min(),
        "alive": int(np.sum(cells == ALIVE)),
        "necrotic": int(np.sum(cells == NECROTIC)),
    }


def add_growth_rate(history):
    for i, h in enumerate(history):
        if i < 24:
            h["growth_rate"] = np.nan
        else:
            h["growth_rate"] = h["radius"] - history[i - 24]["radius"]


def run_simulation(hours=HOURS, seed=1, snapshot_hours=(), verbose=False):
    rng = np.random.default_rng(seed)
    cells = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    middle = GRID_SIZE // 2
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r - middle) ** 2 + (c - middle) ** 2 <= 3**2:
                cells[r, c] = ALIVE

    oxygen = np.full(cells.shape, O2_MEDIUM)
    history, snapshots = [], {}

    for hour in range(hours + 1):
        oxygen = solve_oxygen(cells, oxygen)

        stats = measure(cells, oxygen)
        stats["hour"] = hour
        history.append(stats)
        if hour in snapshot_hours:
            snapshots[hour] = (cells.copy(), oxygen.copy())
        if verbose and hour % 100 == 0:
            print(f"  hour {hour:4d}: radius {stats['radius']:4.0f} um, "
                  f"core radius {stats['core_radius']:4.0f} um, "
                  f"lowest oxygen {stats['lowest_oxygen']:5.1f} mmHg")
        if hour == hours:
            break

        alive_rows, alive_cols = np.nonzero(cells == ALIVE)
        order = rng.permutation(len(alive_rows))
        for i in order:
            r, c = alive_rows[i], alive_cols[i]
            o2 = oxygen[r, c]

            if o2 < O2_DEATH and rng.random() < HYPOXIC_DEATH_CHANCE:
                cells[r, c] = NECROTIC
                continue

            if o2 > O2_DIVIDE and rng.random() < 1.0 / CELL_CYCLE_HOURS:
                spot = random_empty_neighbor(cells, r, c, rng)
                if spot is not None:
                    cells[spot] = ALIVE

        if np.any(cells[[0, 1, -2, -1], :] != EMPTY) or np.any(cells[:, [0, 1, -2, -1]] != EMPTY):
            print("  spheroid reached the edge of the grid, stopping early")
            break

    add_growth_rate(history)
    return {"history": history, "snapshots": snapshots, "final": (cells, oxygen)}


CELL_COLORS = ListedColormap(["white", "#d9534f", "#3a3a3a"])


if __name__ == "__main__":
    WATCH_DAYS = (2, 5, 8, 11, 14, 17)
    SHOW_WINDOWS = True
    print("Growing the spheroid...")
    run = run_simulation(verbose=True, snapshot_hours=[d * 24 for d in WATCH_DAYS])
    history = run["history"]
    last_day = history[-1]["hour"] / 24
    size_mm = GRID_SIZE * DX / 1000
    extent = [0, size_mm, 0, size_mm]

    columns = ["hour", "radius", "core_radius", "rim_thickness", "growth_rate",
               "average_oxygen", "lowest_oxygen", "alive", "necrotic"]
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for h in history:
            writer.writerow([h[c] if isinstance(h[c], int) else round(float(h[c]), 2)
                             for c in columns])

    days = [h["hour"] / 24 for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(days, [h["radius"] for h in history], color="#d9534f")
    axes[0, 0].set_title("Tumor radius")
    axes[0, 0].set_ylabel("um")
    axes[0, 1].plot(days, [h["core_radius"] for h in history], color="#3a3a3a")
    axes[0, 1].set_title("Necrotic core radius")
    axes[0, 1].set_ylabel("um")
    axes[1, 0].plot(days, [h["average_oxygen"] for h in history], label="average in tumor")
    axes[1, 0].plot(days, [h["lowest_oxygen"] for h in history], label="lowest (center)")
    axes[1, 0].axhline(O2_DIVIDE, color="grey", linestyle=":", label="cells stop dividing")
    axes[1, 0].set_title("Oxygen")
    axes[1, 0].set_ylabel("mmHg")
    axes[1, 0].legend()
    axes[1, 1].plot(days, [h["growth_rate"] for h in history], color="#2a7a3a")
    axes[1, 1].set_title("Growth rate (radius change per day)")
    axes[1, 1].set_ylabel("um per day")
    axes[1, 1].set_ylim(bottom=0)
    for ax in axes.flat:
        ax.set_xlabel("time (days)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("tracked_variables.png", dpi=120)

    cells, oxygen = run["final"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(cells, cmap=CELL_COLORS, vmin=0, vmax=2, extent=extent)
    axes[0].set_title(f"Day {last_day:.0f}: cells (red = alive, black = dead)")
    im = axes[1].imshow(np.where(cells != EMPTY, oxygen, np.nan), cmap="Blues_r",
                        vmin=0, vmax=O2_MEDIUM, extent=extent)
    axes[1].set_title(f"Day {last_day:.0f}: oxygen (mmHg)")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("mm")
    fig.tight_layout()
    fig.savefig("snapshot.png", dpi=120)

    fig, axes = plt.subplots(1, len(WATCH_DAYS), figsize=(3 * len(WATCH_DAYS), 3.4))
    for ax, day in zip(axes, WATCH_DAYS):
        ax.imshow(run["snapshots"][day * 24][0], cmap=CELL_COLORS, vmin=0, vmax=2, extent=extent)
        ax.set_title(f"day {day}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Spheroid over time (red = alive, black = dead)")
    fig.tight_layout()
    fig.savefig("over_time.png", dpi=120)

    first = next((h for h in history if h["necrotic"] > 0), None)
    if first:
        print(f"Necrotic core first appears on day {first['hour'] / 24:.1f}, "
              f"when the tumor radius is {first['radius']:.0f} um")
    end = history[-1]
    print(f"Day {last_day:.0f}: tumor radius {end['radius']:.0f} um, "
          f"core radius {end['core_radius']:.0f} um, "
          f"living rim {end['rim_thickness']:.0f} um, "
          f"growth rate {end['growth_rate']:.0f} um per day")
    print("Saved results.csv, tracked_variables.png, snapshot.png and over_time.png")

    if SHOW_WINDOWS:
        print("Showing the plots. Close the windows to finish.")
        plt.show()
