"""
toxin_experiment.py - the main experiment.

Question: if dead cells release a toxin, how much bigger does the dead core get?

We try every combination of:
  * how much toxin each dead cell releases (RELEASE_RATES)
  * how fast the toxin breaks down (HALF_LIVES, in minutes)
and measure, for each one, the average size of the dead core and the
thickness of the living rim over the last few days of the run.

The core size jumps around a bit when toxin is on, so we average over
days 14-16 instead of using a single moment.

Run it with:
    python toxin_experiment.py
It takes a few minutes. Results are saved to results.csv and two plots.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import spheroid

RELEASE_RATES = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]  # units per second
HALF_LIVES = [2, 5, 10, 20, 40]                                  # minutes
RUN_HOURS = 16 * 24
AVERAGE_FROM_HOUR = 14 * 24
SEEDS = [1, 2]   # repeat each setting with 2 different random seeds and average


def toxin_reach(half_life_min):
    """How far toxin typically travels before breaking down: sqrt(D / k), in um."""
    decay_rate = np.log(2) / (half_life_min * 60)
    return np.sqrt(spheroid.TOXIN_DIFFUSION / decay_rate)


def run_one(release_rate, half_life_min):
    core_fracs, rims = [], []
    for seed in SEEDS:
        result = spheroid.run_simulation(release_rate, half_life_min, hours=RUN_HOURS, seed=seed)
        late = [h for h in result["history"] if h["hour"] >= AVERAGE_FROM_HOUR]
        core_fracs += [h["core_diameter"] / h["diameter"] for h in late]
        rims += [h["rim_thickness"] for h in late]
    return np.mean(core_fracs), np.mean(rims)


if __name__ == "__main__":
    core_table = np.zeros((len(HALF_LIVES), len(RELEASE_RATES)))
    rim_table = np.zeros_like(core_table)
    rows = []
    # With no toxin, the half-life doesn't matter, so only run that once
    no_toxin = run_one(0.0, HALF_LIVES[0])

    for i, half_life in enumerate(HALF_LIVES):
        for j, rate in enumerate(RELEASE_RATES):
            core_frac, rim = no_toxin if rate == 0 else run_one(rate, half_life)
            core_table[i, j], rim_table[i, j] = core_frac, rim
            rows.append([rate, half_life, round(toxin_reach(half_life), 1),
                         round(core_frac, 3), round(rim, 1)])
            print(f"release {rate:<7} half-life {half_life:>2} min: "
                  f"core = {core_frac:.0%} of spheroid, living rim = {rim:.0f} um")

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["release_rate", "half_life_min", "toxin_reach_um",
                         "core_fraction", "rim_thickness_um"])
        writer.writerows(rows)

    # Plot 1: heat map of dead core size for every combination
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(core_table, cmap="Greys", vmin=0, vmax=1, origin="lower", aspect="auto")
    ax.set_xticks(range(len(RELEASE_RATES)), [str(r) for r in RELEASE_RATES])
    ax.set_yticks(range(len(HALF_LIVES)),
                  [f"{h} min ({toxin_reach(h):.0f} um)" for h in HALF_LIVES])
    ax.set_xlabel("toxin released per dead cell (units per second)")
    ax.set_ylabel("toxin half-life (how far it travels)")
    ax.set_title("Dead core diameter as a fraction of spheroid diameter (days 14-16)")
    for i in range(len(HALF_LIVES)):
        for j in range(len(RELEASE_RATES)):
            value = core_table[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > 0.6 else "black")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig("core_heatmap.png", dpi=120)

    # Plot 2: living rim thickness vs release rate, one line per half-life
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(RELEASE_RATES))
    for i, half_life in enumerate(HALF_LIVES):
        ax.plot(x, rim_table[i], marker="o",
                label=f"half-life {half_life} min (reach {toxin_reach(half_life):.0f} um)")
    ax.set_xticks(x, [str(r) for r in RELEASE_RATES])
    ax.axhline(rim_table[0, 0], color="grey", linestyle=":", label="no toxin")
    ax.set_xlabel("toxin released per dead cell (units per second)")
    ax.set_ylabel("living rim thickness (um)")
    ax.set_title("How toxin changes the living outer layer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("rim_vs_toxin.png", dpi=120)
    print("Saved results.csv, core_heatmap.png and rim_vs_toxin.png")
