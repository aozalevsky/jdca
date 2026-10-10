# The windowed map for long chains

The whole-chain map needs memory that grows with L². On the deep (Cholesky) route jdca holds three D × D float32 matrices at its
peak, D = 20L; on a 16 GB GPU that stops fitting at about L = 1,500 (see Limits in the README). jdca then computes the map on the CPU
instead, which is correct but slow.

`window=w` (`--window w`) avoids the whole-chain inverse:

* The chain is covered by overlapping tiles of 2w + 1 columns, starting every w columns.
* The sequence weights come from the whole alignment and, in the default mode, a tile's covariance is exactly the matching sub-block
  of the whole-chain covariance, so only the inverse is local. (The adaptive mode recomputes its pseudocount for the tile's width.)
* Each tile's scores get APC on their own; a pair is then read from the tile centred nearest to it.
* The GPU memory of the inverse no longer depends on L (the output map is still L × L). In return only pairs with |i − j| ≤ w are
  scored (the rest are NaN and left out of rankings). A chain no longer than 2w + 1 gets the whole-chain map.

## Example: a long chain with a deep alignment

Integrin α-2 (UniProt P17301, human; L = 1,181), with the 17,216-sequence MSA AFDB provides for it, in `adaptive` mode (which keeps
every sequence). The whole-chain map takes the deep route and needs 8.0 GB of GPU memory at its peak, which still fits on the 16 GB
RTX A4000. The windowed map's inverse needs memory for one tile only:

| | time per map | rank correlation with the whole-chain map, \|i−j\| ≤ 12 / ≤ 100 | top-L/5 precision, \|i−j\| ≥ 6 | top-L/5 precision, \|i−j\| ≥ 24 |
|---|---|---|---|---|
| whole-chain map (GPU) | 5.8 s | — | 0.23 | 0.26 |
| windowed map, w = 100 (GPU) | 1.5 s | 0.86 / 0.73 | 0.37 | 0.29 |

Both times are for one map in a process that has already computed one, from the alignment in memory to the scores (the fastest of
three; the "batch" of [comparison.md](comparison.md)).
Precision is measured as in [benchmark.md](benchmark.md): Cβ–Cβ < 8 Å in the AFDB model, counting pairs whose two residues have
pLDDT ≥ 70. Both maps are scored on the same pairs, |i − j| ≤ 100.

## How close it is to the whole-chain map

Six AlphaFold DB entries, window w = 100. The precision columns show whole-chain map → windowed map, both scored on the pairs with
|i − j| ≤ 100.

| UniProt | protein | L | mode | M | Meff / L | rank correlation, \|i−j\| ≤ 12 / ≤ 100 | top-L/5 precision \|i−j\| ≥ 6 | top-L/5 precision \|i−j\| ≥ 24 |
|---|---|---|---|---|---|---|---|---|
| P0A9S1 | lactaldehyde reductase | 382 | auto | 17,507 | 6.67 | 0.98 / 0.95 | 0.88 → 0.87 | 0.88 → 0.92 |
| | | | adaptive | 17,508 | 20.15 | 0.98 / 0.94 | 0.88 → 0.86 | 0.87 → 0.84 |
| P57740 | Nup107 | 925 | auto | 1,647 | 0.35 | 0.85 / 0.79 | 0.36 → 0.40 | 0.23 → 0.23 |
| | | | adaptive | 1,883 | 0.57 | 0.87 / 0.81 | 0.32 → 0.39 | 0.21 → 0.23 |
| P49588 | alanine–tRNA ligase | 968 | auto | 16,500 | 2.18 | 0.92 / 0.80 | 0.70 → 0.72 | 0.71 → 0.64 |
| | | | adaptive | 17,536 | 6.42 | 0.91 / 0.76 | 0.72 → 0.76 | 0.76 → 0.69 |
| P53396 | ATP-citrate synthase | 1,101 | auto | 14,906 | 0.45 | 0.84 / 0.76 | 0.49 → 0.43 | 0.38 → 0.34 |
| | | | adaptive | 15,581 | 0.28 | 0.81 / 0.73 | 0.45 → 0.40 | 0.39 → 0.33 |
| P29474 | nitric oxide synthase 3 | 1,203 | auto | 11,626 | 0.55 | 0.79 / 0.74 | 0.59 → 0.55 | 0.45 → 0.39 |
| | | | adaptive | 16,793 | 0.66 | 0.73 / 0.71 | 0.62 → 0.51 | 0.45 → 0.39 |
| P17301 | integrin α-2 | 1,181 | auto | 9,649 | 0.94 | 0.89 / 0.77 | 0.23 → 0.36 | 0.28 → 0.27 |
| | | | adaptive | 17,216 | 0.92 | 0.86 / 0.73 | 0.23 → 0.37 | 0.26 → 0.29 |

* The windowed map tracks the whole-chain map closely at short separations. Agreement is weaker further out, where a pair's
  couplings to residues outside its tile are missing.
* Its top-L/5 precision differs from the whole-chain map's by −0.10 to +0.14 across the twelve rows: higher in some, lower in
  others.
* On the four chains of 968–1,203 residues it takes 1.1–1.6 s per map on the GPU, against 1.9–5.8 s for the whole-chain map.
* Use it when the whole-chain map does not fit. It never scores pairs more than w apart.

## Reproduce

```
# bench/data/<UniProt>/ : the AlphaFold DB model and MSA of each target,
#   https://alphafold.ebi.ac.uk/files/AF-<id>-F1-model_v6.cif
#   https://alphafold.ebi.ac.uk/files/msa/AF-<id>-F1-msa_v6.a3m
for u in P0A9S1 P57740 P49588 P53396 P29474 P17301; do
  python bench/guard.py --json bench/out/windowed.json --key $u -- python bench/windowed.py $u --window 100
done
```

The numbers are in `bench/out/windowed.json`. The times were measured as in [benchmark.md](benchmark.md): one target at a time, kept
only if no other program used the GPU meanwhile (`bench/guard.py`; its verdict is stored as `guard` per target).

```python
import jdca
r = jdca.gaussdca('alignment.a3m', mode='adaptive', window=100)
```
