# Necrotic core simulation

A 2D simulation of a tumor spheroid. Oxygen spreads in from the outside, cells use it up, and cells in the middle die when it runs out. On top of that, dead cells can leak a toxin that harms the living cells around them.

## How to run it

You need Python 3 with NumPy, SciPy and Matplotlib:

```
pip install -r requirements.txt
```

Then, from this folder:

```
python spheroid.py           # one spheroid with and without toxin (about 15 seconds)
python toxin_experiment.py   # the full experiment (about 3 minutes)
```

The plots are saved as PNG files in the same folder. The `example-results` folder has the plots from a run I already did, so you can see what to expect.

## Files

- `spheroid.py` is the model: the grid, oxygen, toxin, and the cell rules. All the settings are at the top.
- `toxin_experiment.py` changes the toxin release rate and half-life, then measures the dead core.

## What the example results show

**The basic model matches real spheroids.** With no toxin, the dead core first appears when the spheroid is about 500 µm across, and the living rim stays about 190–200 µm thick. Real spheroids get a dead core at about 400–500 µm with a rim of about 150–200 µm, so the oxygen model is in the right range.

**Toxin does nothing until it suddenly does a lot.** For low release rates the dead core stays the same size as with no toxin (61% of the spheroid's diameter). Past a certain release rate, the core jumps to 75–90% of the diameter and the living rim shrinks from about 200 µm to as little as 40–90 µm (`core_heatmap.png`, `rim_vs_toxin.png`).

**Toxin that lasts longer needs less release to cause damage.** With a 2 minute half-life, it takes a release of 0.02 to see any change. With a 40 minute half-life, 0.002–0.005 is already enough. The toxin's reach, √(D/k), is shown next to each half-life on the plots.

**The core grows in sudden jumps.** In `toxin_over_time.png` and `growth.png`, the dead core with toxin stays the same size for days, then a wave of death spreads outward in less than a day. One death releases toxin, which kills the neighbors, which release more toxin.

At the highest release rates the numbers bounce around a bit (for example 0.87 then 0.78). That's because of when the death waves happen to fall within the days 14–16 averaging window, not a real effect. Adding more seeds to `SEEDS` would smooth it out.

## Where the numbers come from

Please check these against the papers yourself before your write-up. Some of them are rounded.

| Setting | Value | Source / reasoning |
|---|---|---|
| Square size | 20 µm | About the size of one cell |
| Oxygen diffusion | 2000 µm²/s (2×10⁻⁵ cm²/s) | Standard value for oxygen in tissue, e.g. Grimes et al. 2014, *J. R. Soc. Interface* |
| Oxygen use | 15 mmHg/s | Converted from about 0.02 mol/m³/s, the range Grimes et al. 2014 report for spheroids |
| Oxygen in medium | 100 mmHg | Rough value; it's lower than the 160 mmHg of air because the medium sits on top of the cells |
| Stop dividing below | 10 mmHg | Usual cutoff for "hypoxic" |
| Start dying below | 1 mmHg | Near-zero oxygen ("anoxic") |
| Cell cycle | 24 hours | Typical cancer cell line |
| Toxin diffusion | 10 µm²/s | Protein-sized molecules move much slower than oxygen |
| Toxin release, half-life | swept | These are what the experiment tests |
| Death chances, weakness 0.5, 24 h leak time | guesses | Model choices; a good thing to test in a sensitivity check |

Background papers: Greenspan 1972 (*Studies in Applied Mathematics*) had an inhibitor made in the dead core. Freyer 1988 (*Cancer Research*) studied how necrosis limits spheroid growth. Sutherland 1988 (*Science*) is a good review of spheroid structure.

## What the model can't capture

- It's 2D, while real spheroids are 3D balls. In 3D, oxygen gets in from more directions, which changes the numbers a bit.
- Cells can only divide into an empty square next to them, so only the outer layer grows. Real cells push each other around.
- Dead cells never shrink or get cleared away.
- Real dying cells release many different molecules, not one "toxin", and the kill threshold of 1 is just a unit.

## Ideas for next steps

1. Run a sensitivity check: change the guessed settings one at a time by ±50% and see if the jump still happens.
2. Try more release rates between 0.002 and 0.01 to pin down exactly where the jump is.
3. Plot the threshold release rate against the toxin's reach to see if they follow a simple pattern.
