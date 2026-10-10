# The adaptive mode

`jdca.gaussdca(..., mode='adaptive')` (`jdca ... --mode adaptive`) is a second setting of the GaussDCA estimator:

| | default (`auto`, = GaussDCA.jl) | `adaptive` |
|---|---|---|
| pseudocount | 0.8 | 0.5 when M > 20 L, else max(0.5, 0.95 − 0.45 M / (20 L)) |
| reweighting neighbours | Hamming distance < ⌊θL⌋, θ = auto | identity ≥ 0.8 L |
| diagonal | — | + 1e-4 |
| sequence filter | > 90% gaps dropped | none |
| APC | means over the off-diagonal entries | means over all entries |

## On the benchmark targets

The three targets of [benchmark.md](benchmark.md), with the same precision metric: Cβ–Cβ < 8 Å in the AFDB model, counting pairs
whose two residues have pLDDT ≥ 70.

| | mode | \|i−j\| ≥ 6: top L/5 | L/2 | L | \|i−j\| ≥ 24: top L/5 | L/2 | L | pseudocount | Meff | GPU, batch (per map) |
|---|---|---|---|---|---|---|---|---|---|---|
| P34884 | default | 0.57 | 0.49 | 0.42 | 0.61 | 0.47 | 0.41 | 0.8 | 620.5 | 0.036 s |
| | adaptive | 0.70 | 0.47 | 0.38 | 0.61 | 0.47 | 0.41 | 0.50 | 1,601.9 | 0.029 s |
| P0A9S1 | default | 0.89 | 0.81 | 0.71 | 0.91 | 0.78 | 0.69 | 0.8 | 2,546.1 | 0.59 s |
| | adaptive | 0.89 | 0.76 | 0.68 | 0.88 | 0.79 | 0.69 | 0.50 | 7,698.8 | 0.52 s |
| P57740 | default | 0.35 | 0.21 | 0.14 | 0.22 | 0.12 | 0.08 | 0.8 | 322.4 | 0.28 s |
| | adaptive | 0.32 | 0.19 | 0.15 | 0.19 | 0.13 | 0.10 | 0.90 | 525.8 | 0.29 s |

The time is per map in a running process ("batch", as in [benchmark.md](benchmark.md)). Three targets are too few to rank the
two settings.
