"""
spheroid.py - a 2D simulation of a tumor spheroid with a necrotic (dead) core.

How the model works
-------------------
* The world is a square grid. Each square is 20 micrometers wide, about the size
  of one cell. A square is either EMPTY (culture medium), ALIVE, or NECROTIC (dead).
* Oxygen comes from the medium and spreads (diffuses) into the spheroid.
  Every living cell uses some up, so the middle gets less oxygen than the edge.
* Each time step is 1 hour. In each step:
    1. Work out the oxygen level in every square.
    2. Work out the toxin level (dead cells leak toxin for a while after dying).
    3. Living cells with enough oxygen may divide into an empty neighbor square.
    4. Living cells with almost no oxygen may die.
    5. Living cells with too much toxin may die. Cells that are already short
       on oxygen are weaker, so they die at a lower toxin level.
* Setting RELEASE_RATE = 0 switches the toxin off, which gives the normal
  "oxygen only" model to compare against.

Oxygen and toxin spread in seconds, but cells change over hours, so each hour we
solve for the "settled" (steady state) oxygen and toxin levels. See solve_field().

Run this file directly to simulate one spheroid and save plots:
    python spheroid.py
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------------------
# Parameters. Values marked [lit] are taken from (or close to) published papers;
# see README.md for the sources. Values marked [guess] are model choices.
# ----------------------------------------------------------------------------

GRID_SIZE = 100             # grid is 100 x 100 squares (2 mm x 2 mm)
DX = 20.0                   # width of one square, in micrometers (um) [lit: ~1 cell]
HOURS = 480                 # how long to run (480 h = 20 days)

# Oxygen (measured in mmHg, the usual unit for oxygen in tissue)
O2_MEDIUM = 100.0           # oxygen level in the medium around the spheroid [lit]
O2_DIFFUSION = 2000.0       # how fast oxygen spreads, um^2 per second [lit]
O2_USE = 15.0               # max oxygen use by living cells, mmHg per second [lit]
O2_HALF = 1.0               # cells use oxygen at half speed when O2 = this [lit]
O2_DIVIDE = 10.0            # cells only divide above this oxygen level [lit]
O2_DEATH = 1.0              # cells start dying below this oxygen level [lit]

# Cells
CELL_CYCLE_HOURS = 24.0     # a well-fed cell divides about once a day [lit]
HYPOXIC_DEATH_CHANCE = 0.2  # chance per hour that a starving cell dies [guess]

# Toxin from dead cells (toxin is measured in "kill units": 1.0 = enough to kill)
RELEASE_RATE = 0.0          # toxin made per dead cell, units per second (0 = off)
TOXIN_HALF_LIFE_MIN = 10.0  # how long until half the toxin has broken down, minutes
TOXIN_DIFFUSION = 10.0      # how fast toxin spreads, um^2 per second [lit: proteins]
RELEASE_HOURS = 24          # dead cells only leak toxin for this long [guess]
TOXIN_KILL_LEVEL = 1.0      # toxin level that kills a healthy cell
HYPOXIC_WEAKNESS = 0.5      # a starving cell dies at only 50% of the kill level [guess]
TOXIN_DEATH_CHANCE = 0.5    # chance per hour that a poisoned cell dies [guess]

# Cell states
EMPTY, ALIVE, NECROTIC = 0, 1, 2


def solve_field(inside, diffusion, uptake, production, outside_value):
    """
    Find the settled (steady state) level of a chemical in every square.

    In each square inside the spheroid, balance is reached when:
        (what diffuses in from the 4 neighbors) - (what is used up) + (what is made) = 0

    Written out for square i:
        diffusion / DX^2 * (sum of the 4 neighbors - 4 * c_i) - uptake_i * c_i + production_i = 0

    Squares outside the spheroid are medium, so their level is fixed at
    outside_value. That gives one equation per inside square, and SciPy solves
    them all at once (a "sparse linear system").

    inside      - True/False grid, which squares to solve for
    diffusion   - how fast the chemical spreads (um^2 per second)
    uptake      - grid, how fast the chemical is removed in each square (per second)
    production  - grid, how fast the chemical is made in each square (per second)
    """
    field = np.full(inside.shape, outside_value, dtype=float)
    rows, cols = np.nonzero(inside)
    n = len(rows)
    if n == 0:
        return field

    # Give each inside square a number 0, 1, 2, ... so it can be a row of the matrix
    number = -np.ones(inside.shape, dtype=int)
    number[rows, cols] = np.arange(n)

    a = diffusion / DX**2
    diagonal = -4 * a - uptake[rows, cols]
    right_side = -production[rows, cols]
    matrix_rows, matrix_cols, matrix_vals = [], [], []

    for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        n_rows, n_cols = rows + d_row, cols + d_col
        neighbor_inside = inside[n_rows, n_cols]
        # Neighbor is also inside: it is another unknown in the equation
        matrix_rows.append(np.arange(n)[neighbor_inside])
        matrix_cols.append(number[n_rows, n_cols][neighbor_inside])
        matrix_vals.append(np.full(neighbor_inside.sum(), a))
        # Neighbor is medium: its value is known, so move it to the right side
        right_side = right_side - a * outside_value * (~neighbor_inside)

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
    """
    Oxygen level everywhere. Cells use oxygen more slowly when there is very
    little left (Michaelis-Menten uptake: use = O2_USE * O2 / (O2 + O2_HALF)).
    That makes the equation non-linear, so we repeat the solve a few times,
    each time using the last answer to set the uptake rate.
    """
    inside = cells != EMPTY
    oxygen = oxygen_guess
    for _ in range(4):
        uptake = np.where(cells == ALIVE, O2_USE / (np.maximum(oxygen, 0) + O2_HALF), 0.0)
        oxygen = solve_field(inside, O2_DIFFUSION, uptake, np.zeros(cells.shape), O2_MEDIUM)
    return np.maximum(oxygen, 0)


def solve_toxin(cells, hours_dead, release_rate, half_life_min):
    """Toxin level everywhere. Only recently dead cells leak toxin."""
    if release_rate == 0:
        return np.zeros(cells.shape)
    inside = cells != EMPTY
    decay_rate = np.log(2) / (half_life_min * 60)   # per second
    leaking = (cells == NECROTIC) & (hours_dead < RELEASE_HOURS)
    production = np.where(leaking, release_rate, 0.0)
    uptake = np.where(inside, decay_rate, 0.0)       # toxin breaks down everywhere
    return solve_field(inside, TOXIN_DIFFUSION, uptake, production, 0.0)


def random_empty_neighbor(cells, row, col, rng):
    """Pick a random empty square among the 8 neighbors, or None if there is none."""
    options = []
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            r, c = row + d_row, col + d_col
            if cells[r, c] == EMPTY:
                options.append((r, c))
    if not options:
        return None
    return options[rng.integers(len(options))]


def measure(cells):
    """Turn cell counts into sizes in micrometers (as if each part were a circle)."""
    area_per_square = DX**2
    total_area = np.sum(cells != EMPTY) * area_per_square
    core_area = np.sum(cells == NECROTIC) * area_per_square
    radius = np.sqrt(total_area / np.pi)
    core_radius = np.sqrt(core_area / np.pi)
    return {
        "diameter": 2 * radius,
        "core_diameter": 2 * core_radius,
        "rim_thickness": radius - core_radius,
        "alive": int(np.sum(cells == ALIVE)),
        "necrotic": int(np.sum(cells == NECROTIC)),
    }


def run_simulation(release_rate=RELEASE_RATE, half_life_min=TOXIN_HALF_LIFE_MIN,
                   hours=HOURS, seed=1, snapshot_hours=(), verbose=False):
    """
    Grow one spheroid from a small seed of cells.

    Returns a dictionary with:
      history   - a list of measurements, one per hour
      snapshots - {hour: (cells, oxygen, toxin)} for each hour in snapshot_hours
      final     - (cells, oxygen, toxin) at the end
    """
    rng = np.random.default_rng(seed)    # same seed -> same result every time
    cells = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    hours_dead = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    # Start with a small blob of living cells in the middle (radius 3 squares)
    middle = GRID_SIZE // 2
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r - middle) ** 2 + (c - middle) ** 2 <= 3**2:
                cells[r, c] = ALIVE

    oxygen = np.full(cells.shape, O2_MEDIUM)
    history, snapshots = [], {}

    for hour in range(hours + 1):
        # 1 and 2: settled oxygen and toxin levels for this hour
        oxygen = solve_oxygen(cells, oxygen)
        toxin = solve_toxin(cells, hours_dead, release_rate, half_life_min)

        stats = measure(cells)
        stats["hour"] = hour
        history.append(stats)
        if hour in snapshot_hours:
            snapshots[hour] = (cells.copy(), oxygen.copy(), toxin.copy())
        if verbose and hour % 100 == 0:
            print(f"  hour {hour:4d}: diameter {stats['diameter']:5.0f} um, "
                  f"core {stats['core_diameter']:5.0f} um")
        if hour == hours:
            break

        hours_dead[cells == NECROTIC] += 1

        # Go through the living cells in a random order so no direction is favored
        alive_rows, alive_cols = np.nonzero(cells == ALIVE)
        order = rng.permutation(len(alive_rows))
        for i in order:
            r, c = alive_rows[i], alive_cols[i]
            o2, tox = oxygen[r, c], toxin[r, c]

            # 4: death from lack of oxygen
            if o2 < O2_DEATH and rng.random() < HYPOXIC_DEATH_CHANCE:
                cells[r, c] = NECROTIC
                hours_dead[r, c] = 0
                continue

            # 5: death from toxin (lower kill level if the cell is starving)
            kill_level = TOXIN_KILL_LEVEL
            if o2 < O2_DIVIDE:
                kill_level *= HYPOXIC_WEAKNESS
            if tox > kill_level and rng.random() < TOXIN_DEATH_CHANCE:
                cells[r, c] = NECROTIC
                hours_dead[r, c] = 0
                continue

            # 3: division, only with enough oxygen and an empty neighbor
            if o2 > O2_DIVIDE and rng.random() < 1.0 / CELL_CYCLE_HOURS:
                spot = random_empty_neighbor(cells, r, c, rng)
                if spot is not None:
                    cells[spot] = ALIVE

        # Stop if the spheroid reaches the edge of the grid
        if np.any(cells[[0, 1, -2, -1], :] != EMPTY) or np.any(cells[:, [0, 1, -2, -1]] != EMPTY):
            print("  spheroid reached the edge of the grid, stopping early")
            break

    return {"history": history, "snapshots": snapshots, "final": (cells, oxygen, toxin)}


# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------

CELL_COLORS = ListedColormap(["white", "#d9534f", "#3a3a3a"])  # empty, alive, dead


def plot_snapshot(ax_row, cells, oxygen, toxin, title):
    """Draw cells, oxygen and toxin side by side on a row of 3 axes."""
    size_mm = GRID_SIZE * DX / 1000
    extent = [0, size_mm, 0, size_mm]
    ax_row[0].imshow(cells, cmap=CELL_COLORS, vmin=0, vmax=2, extent=extent)
    ax_row[0].set_title(f"{title}\ncells (red = alive, black = dead)")
    im = ax_row[1].imshow(np.where(cells != EMPTY, oxygen, np.nan), cmap="Blues_r",
                          vmin=0, vmax=O2_MEDIUM, extent=extent)
    ax_row[1].set_title("oxygen (mmHg)")
    plt.colorbar(im, ax=ax_row[1], fraction=0.046)
    im = ax_row[2].imshow(np.where(cells != EMPTY, toxin, np.nan), cmap="Purples",
                          vmin=0, vmax=max(1.0, np.nanmax(toxin)), extent=extent)
    ax_row[2].set_title("toxin (1 = kills a healthy cell)")
    plt.colorbar(im, ax=ax_row[2], fraction=0.046)
    for ax in ax_row:
        ax.set_xlabel("mm")


def plot_growth(ax, history, label, color):
    hours = [h["hour"] / 24 for h in history]
    ax.plot(hours, [h["diameter"] for h in history], color=color, label=f"{label}: whole spheroid")
    ax.plot(hours, [h["core_diameter"] for h in history], color=color, linestyle="--",
            label=f"{label}: dead core")


if __name__ == "__main__":
    # Compare the normal model (no toxin) with one toxin setting
    TOXIN_EXAMPLE = 0.01   # units per second per dead cell
    WATCH_DAYS = (6, 7, 8, 12, 14, 16)   # days to take pictures of the toxin run
    print("Running the oxygen-only model...")
    base = run_simulation(release_rate=0.0, verbose=True)
    print("Running the model with toxin...")
    tox = run_simulation(release_rate=TOXIN_EXAMPLE, verbose=True,
                         snapshot_hours=[d * 24 for d in WATCH_DAYS])
    last_day = base["history"][-1]["hour"] / 24

    # Figure 1: what the spheroid looks like at the end, with and without toxin
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    plot_snapshot(axes[0], *base["final"], f"No toxin, day {last_day:.0f}")
    plot_snapshot(axes[1], *tox["final"], f"With toxin, day {last_day:.0f}")
    fig.tight_layout()
    fig.savefig("snapshot.png", dpi=120)

    # Figure 2: growth over time
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_growth(ax, base["history"], "no toxin", "#1f77b4")
    plot_growth(ax, tox["history"], "with toxin", "#e8710a")
    ax.set_xlabel("time (days)")
    ax.set_ylabel("diameter (um)")
    ax.set_title("Spheroid growth and dead core size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("growth.png", dpi=120)

    # Figure 3: the toxin run over time, to see how the dead core spreads
    fig, axes = plt.subplots(1, len(WATCH_DAYS), figsize=(3 * len(WATCH_DAYS), 3.4))
    size_mm = GRID_SIZE * DX / 1000
    for ax, day in zip(axes, WATCH_DAYS):
        cells = tox["snapshots"][day * 24][0]
        ax.imshow(cells, cmap=CELL_COLORS, vmin=0, vmax=2, extent=[0, size_mm, 0, size_mm])
        ax.set_title(f"day {day}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("With toxin (red = alive, black = dead)")
    fig.tight_layout()
    fig.savefig("toxin_over_time.png", dpi=120)

    # Check against real spheroids: when does the dead core first show up?
    for name, run in [("no toxin", base), ("with toxin", tox)]:
        first = next((h for h in run["history"] if h["necrotic"] > 0), None)
        end = run["history"][-1]
        if first:
            print(f"{name}: dead core first appears on day {first['hour'] / 24:.1f}, "
                  f"when the spheroid is {first['diameter']:.0f} um across")
        print(f"{name}: at the end, living rim is {end['rim_thickness']:.0f} um thick")
    print("Saved snapshot.png, growth.png and toxin_over_time.png")
