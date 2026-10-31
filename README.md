# Necrotic core simulation

A 2D simulation of a growing tumor spheroid (a small ball of cancer cells). Oxygen spreads in from the outside and the cells use it up. Once the tumor gets too big, oxygen can't reach the middle, so the cells there die and a necrotic (dead) core forms.

The simulation keeps track of four things every hour:

1. **Tumor radius** (µm)
2. **Necrotic core radius** (µm)
3. **Oxygen**: the average level in the tumor and the lowest level, which is at the center (mmHg)
4. **Growth rate**: how much the tumor radius grew over the last 24 hours (µm per day)

## How to run it

It's tested on **Python 3.14**, the newest stable version. It should also work on 3.11 or newer.

```
pip install -r requirements.txt
python spheroid.py
```

It takes about 10 seconds and saves these files in the folder you run it from:

- `results.csv`: every hour's numbers, which you can open in Excel or Google Sheets
- `tracked_variables.png`: the four tracked variables over time
- `snapshot.png`: the tumor and its oxygen levels on the last day
- `over_time.png`: pictures of the tumor on different days

The `example-results` folder has these files from a run I already did.

## How the model works

- The grid is 100 × 100 squares, and each square is 20 µm wide, about the size of one cell. A square can be empty (medium), alive, or dead.
- Each time step is 1 hour. In each step:
  1. The oxygen level is worked out for every square. Oxygen spreads in from the medium, and living cells use it up.
  2. Living cells with enough oxygen (above 10 mmHg) can divide into an empty square next to them, about once a day.
  3. Living cells with almost no oxygen (below 1 mmHg) can die.
- All the settings are at the top of `spheroid.py`, with comments.

## What the example results show

- The tumor radius grows steadily, at about 25–40 µm per day.
- The oxygen at the center drops as the tumor grows. It reaches almost zero on about day 6.
- The necrotic core first appears on day 6.8, when the tumor radius is about 250 µm (500 µm across). Real spheroids get a dead core at about 400–500 µm across, so this is in the right range.
- After that, the core grows at about the same speed as the tumor. The living rim stays about 190–200 µm thick. Real spheroids keep a living rim of about 150–200 µm.

## Where the numbers come from

Please check these against the papers yourself before your write-up. Some of them are rounded.

| Setting | Value | Source / reasoning |
|---|---|---|
| Square size | 20 µm | About the size of one cell |
| Oxygen diffusion | 2000 µm²/s (2×10⁻⁵ cm²/s) | Standard value for oxygen in tissue, e.g. Grimes et al. 2014, *J. R. Soc. Interface* |
| Oxygen use | 15 mmHg/s | Converted from about 0.02 mol/m³/s, the range Grimes et al. 2014 report for spheroids |
| Oxygen in medium | 100 mmHg | Rough value. It's lower than the 160 mmHg in air because the medium sits on top of the cells |
| Stop dividing below | 10 mmHg | Usual cutoff for "hypoxic" (low oxygen) |
| Start dying below | 1 mmHg | Near-zero oxygen ("anoxic") |
| Cell cycle | 24 hours | Typical cancer cell line |
| Death chance per hour | 0.2 | Guess |

Background: Sutherland 1988 (*Science*) is a good review of how spheroids are built, and it includes the size where a dead core appears.

## What the model can't capture

- It's 2D, while real spheroids are 3D balls. In 3D, oxygen comes in from more directions, so the numbers change a bit.
- Cells can only divide into an empty square next to them, so only the outer layer grows. Real cells push each other around.
- Dead cells never shrink or get cleared away, so the tumor never stops growing. Real spheroids slow down and level off.
- It only tracks oxygen. It leaves out glucose and other nutrients.
