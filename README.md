# jdca — GaussDCA for fast GPU inference

jdca is a re-implementation of **GaussDCA**, the multivariate Gaussian direct-coupling analysis of Baldassi, Zamparo, Feinauer,
Procaccini, Zecchina, Weigt and Pagnani (*PLoS ONE* 2014), which predicts residue–residue contacts from a multiple sequence
alignment. The method and its reference implementation, [GaussDCA.jl](https://github.com/carlobaldassi/GaussDCA.jl), are their
work. jdca keeps their estimator and rearranges the arithmetic for fast inference on GPUs (float32 with JAX) and CPUs (BLAS/LAPACK).
By default it computes exactly GaussDCA.jl's estimator, with the same settings and output format.

* **Same answer.** jdca's scores equal a float64 transcription of GaussDCA.jl to float32 precision, and the reweighting threshold θ
  and Meff agree with GaussDCA.jl's to the last printed digit. The side-by-side comparison with GaussDCA.jl is in
  [docs/benchmark.md](docs/benchmark.md).
* **Fast.** In a running process, one coupling map of the three benchmark targets takes from a few hundredths of a second to under
  a second on one GPU: 10–170× less than GaussDCA.jl on 8 CPU threads. A one-off `jdca` command adds about 1.5 s of GPU start-up,
  which the optional server removes (see [How long a map takes](#how-long-a-map-takes)).

If you use jdca, please cite the GaussDCA paper ([Citing](#citing)).

[Add-ons](#add-ons), all opt-in, extend it: a resident GPU server, an adaptive setting, a windowed map for long chains, a ridge DCA,
and the GPU DCA used by Humphreys et al., with the settings of its published script.

## The default: GaussDCA.jl's estimator

| | GaussDCA.jl `gDCA` | `jdca.gaussdca` |
|---|---|---|
| alphabet | 20 amino acids; gap and any other symbol = state 21; lowercase / `.` dropped | same |
| states | q = the largest state present (21 whenever there is a gap or another symbol); the last is the reference state | same |
| sequence filter | drop sequences with > 90% gaps (`max_gap_fraction=0.9`) | same |
| reweighting | weight = 1 / #sequences at Hamming distance < ⌊θL⌋; θ = `:auto` = min(0.5, 0.1216 / mean identity) | same, computed exactly |
| pseudocount | 0.8, mixing with the uniform distribution | same |
| couplings | −C⁻¹ of the pseudocounted covariance of the q − 1 non-reference states | same |
| score | Frobenius norm of the zero-sum-gauged (q − 1) × (q − 1) blocks, APC | same |
| output | pairs with \|i − j\| ≥ 5, best first, `i j score` | same format (`jdca` command) |

GaussDCA.jl's other settings (pseudocount, θ, max_gap_fraction) are arguments of `jdca.gaussdca` and options of `jdca`. Its `:DI`
score is not implemented.

## Where the speed comes from

The estimator is GaussDCA's; only the order and precision of the arithmetic change.

1. **Exact identities.** Pairwise sequence identities are integer counts, and jdca computes them exactly:
   * on the GPU, as int8 one-hot dot products with int32 accumulation, on the tensor cores;
   * on the CPU, on sequences bit-packed in 5-bit fields compared with XOR. The per-field mismatch flags are summed in place
     without popcount, the approach DCAUtils.jl (Carlo Baldassi) takes.

   Either way jdca visits only the upper triangle of row-block pairs, and skips block pairs whose gap counts already rule out the
   identity threshold. The θ = auto statistic (the mean identity over all pairs) uses the same kernels.
2. **No cancellation in the covariance.** The pseudocounted covariance `Pij − Pi Piᵀ` is exactly the covariance of a mixture:
   (1 − pc) × the weighted alignment plus pc × independent uniform columns. So `C = A + RᵀR`, with `A` block-diagonal and known in
   closed form, and `R` the centred, weighted one-hot rows plus one mean-difference row. In float32 this avoids subtracting two
   nearly equal terms.
3. **The cheaper inverse.** With r = M + 1 rows and D = 20L columns:
   * a shallow alignment (r < D/√2) uses the Woodbury identity, `C⁻¹ = A⁻¹ − YᵀY` with `Y = chol(I + R A⁻¹ Rᵀ)⁻¹ R A⁻¹`, so only an
     r × r matrix is factorised;
   * a deep one uses Cholesky, `C⁻¹ = ZᵀZ` with `Z = chol(C)⁻¹`.

   Only the 20 × 20 block norms of `C⁻¹` are formed, tile by tile and over the upper triangle only. On the deep route `Z` is
   triangular, so its all-zero row chunks are skipped.
4. **float32 that agrees with float64.** One long float32 contraction on a GPU is far less accurate than LAPACK's blocked sums.
   The contractions over sequences, the deep route's Gram matrix and the tile products, are therefore accumulated in 256-row
   chunks, which matches LAPACK's accuracy at the same speed.
5. **Compile once.** Shapes are padded to buckets of 8 steps per octave (in L, and in M on the Woodbury route; multiples of 2048 on
   the Cholesky route). The pseudocount and thresholds are run-time arguments, so one ahead-of-time compiled XLA program serves every
   alignment in a bucket. The GPU programs are kept in a persistent cache; CPU programs are compiled afresh in each process. Unless
   the program has set up its own, jdca turns on JAX's persistent compilation cache in `~/.cache/jdca/jax` (or `$JDCA_JAX_CACHE`)
   the first time it uses JAX, and it then applies to the program's other JAX code too.

On the CPU the same factorisation runs through BLAS/LAPACK (`ssyrk`, `spotrf`/`spotri`, or the Woodbury route); the identity
counts are compiled XLA kernels on both devices.

## Install

```
pip install -e .              # CPU
pip install -e ".[cuda]"      # with JAX for CUDA 12
```

Requires Python ≥ 3.10, NumPy, SciPy and JAX (tested with JAX 0.9.2, and with JAX 0.4.30 on the CPU). The benchmark scripts also need biotite
(`pip install -e ".[bench]"`).

## Use

```
jdca alignment.a3m -o contacts.txt                  # GaussDCA.jl's estimator and defaults
jdca alignment.a3m --theta 0.2 --pseudocount 0.5    # any GaussDCA.jl setting
```

The output has one `i j score` line per pair (1-based, |i − j| ≥ 5, best first), as GaussDCA.jl's `printrank` writes it.

```python
import jdca
r = jdca.gaussdca('alignment.a3m')          # DCA(scores=(L, L) array, meff, theta)
i, j, s = jdca.ranking(r.scores)            # 0-based pairs, best first
```

`alignment` is a FASTA/A3M path or an (M, L) int8 code matrix (`jdca.read_alignment`). A matrix is used as given: it gets no gap
filter, and passing `max_gap_fraction` with one is an error. `device='auto'` uses a running server, else a local GPU, else the CPU;
`'gpu'`, `'cpu'` and `'server'` force one. A GaussDCA map that does not fit in GPU memory is computed on the CPU, with a warning.

## How long a map takes

The benchmarks report three numbers, for three ways of using jdca:

| | what is timed | use it for |
|---|---|---|
| **one-off** | one `jdca` command, from process start to exit | a single alignment from the shell |
| **one-off with `jdca-server`** | the same command while the server runs | many separate commands, e.g. a pipeline that runs `jdca` once per protein |
| **batch** | one map in a process that has already computed one | many alignments from one Python process (`for a in alignments: jdca.gaussdca(a)`) |

A one-off command on the GPU spends about 1.5 s starting Python, JAX and the GPU before its first map. In a batch that is paid once
per process; with the server, once per server. The first alignment of a new size class also loads (or, the very first time,
compiles) its programs. On the benchmark targets a map then costs 0.04–0.6 s on the GPU in a batch, 0.2–0.9 s as a command with
the server running, and 1.7–2.4 s as a one-off command ([docs/benchmark.md](docs/benchmark.md)).

## Add-ons

None of these changes the default.

### Resident GPU server

`jdca-server` keeps one process on the GPU with its programs loaded. Clients (the `jdca` command, `jdca.gaussdca`,
`jdca.ridge_dca`, `jdca.humphreys_dca`) send the int8 alignment over a Unix socket and get back exactly what the same call
computes locally. It helps when many separate processes each compute a map; a single Python process computing many maps does not
need it.

```
jdca-server start      # background; socket and log in $XDG_RUNTIME_DIR/jdca/ (else /tmp/jdca-<uid>/), mode 0700
jdca-server status     # also reports whether the server runs the same jdca sources as this checkout
jdca-server stop
```

On Linux, both ends check that the other runs as the same user. A client compares a fingerprint of the jdca sources and its
Python environment with the server's at connection. If they differ (for example, jdca was edited after the server started), the
client computes locally and says so once. Restart the server after updating jdca. A server that does not answer within 5 s (a
suspended process, say) is treated as absent: the client computes locally, and `jdca-server status` reports the server's pid.

### Adaptive mode

`mode='adaptive'` (`--mode adaptive`) is a second setting of the same estimator: an adaptive pseudocount, a fixed identity threshold
for the reweighting, no gap filter. The settings and its results are in [docs/adaptive.md](docs/adaptive.md).

### Windowed map for long chains

The whole-chain inverse needs memory that grows as L² (see Limits). `window=w` (`--window w`) inverts overlapping tiles of 2w + 1
columns instead:
* The weights come from the whole alignment and, in the default mode, each tile's covariance is exactly the matching sub-block of
  the whole-chain covariance, so only the inverse is local. (The adaptive mode recomputes its pseudocount for the tile's width.)
* Each tile gets APC on its own, and a pair is read from the tile centred nearest to it.
* The GPU memory of the inverse no longer grows with L; only pairs with |i − j| ≤ w are scored (the rest are NaN and left out of
  rankings). A chain no longer than 2w + 1 gets the whole-chain map.

On six AlphaFold DB entries its top-L/5 contact precision is within −0.10 to +0.14 of the whole-chain map's. On the four chains of
968–1,203 residues it takes 1.1–1.6 s per map on the GPU, against 1.9–5.8 s for the whole-chain map; it matters most where the
whole-chain map no longer fits in GPU memory and would be computed on the CPU ([docs/windowed.md](docs/windowed.md)).

### Ridge DCA, and the DCA of Humphreys et al.

Humphreys et al. (*Science* 2021) screened protein pairs with a fast GPU DCA, a ridge-regularised Gaussian model that Zhang,
Humphreys, Pei et al. later published as a TensorFlow script (*Science* 2025; bioRxiv 2024.10.01.615885, supplementary Methods
M5.3). jdca offers it in two forms, both on the same fast path as GaussDCA (Woodbury or Cholesky route, chunked float32, compiled
programs, server):

* **`jdca.humphreys_dca`** (`--method humphreys`) is their estimator with their settings: every sequence kept, weights
  1 / #{sequences with identity > 0.8 L}, the 21-state covariance (the gap is a state) plus one ridge 4.5 / √Meff, and per position
  pair the Frobenius norm of the raw 20 × 20 non-gap block of `C⁻¹`, then APC. On the benchmark targets it reproduces their script's
  scores to 5e-4 or better and shares all of its top-L pairs ([docs/comparison.md](docs/comparison.md)).
* **`jdca.ridge_dca`** (`--method ridge`) is a variant of it: a per-column ridge λᵢ = c / √nᵢ (nᵢ = the column's weighted non-gap
  count, c = 4.07) instead of one ridge, and pairs scored by partial correlations, the Frobenius norm of the 20 × 20 non-gap block of
  `Pᵢⱼ,ₐᵦ / √(Pᵢᵢ,ₐₐ Pⱼⱼ,ᵦᵦ)` (`P = C⁻¹`), then APC. `column_ridge=False`, `score='fn'` and `c` give the intermediate settings.

Neither has a windowed form or a CPU fallback for GPU memory; on the CPU they run the same XLA programs as on the GPU.

Which estimator to use? On seven AlphaFold DB entries, scored with seven contact metrics (six of them at three list lengths),
GaussDCA, its adaptive mode, the estimator of Humphreys et al. and the ridge DCA are within 0.023 of each other on every
precision, recall and coverage mean, and none is consistently best ([docs/comparison.md](docs/comparison.md#other-contact-metrics)).

```
jdca alignment.a3m --mode adaptive                  # adaptive mode
jdca alignment.a3m --window 100                     # windowed map: pairs up to 100 apart, memory independent of L
jdca alignment.a3m --method humphreys               # the DCA of Humphreys et al., their settings
jdca alignment.a3m --method ridge                   # ridge DCA, partial correlations
```

```python
r = jdca.gaussdca('alignment.a3m', mode='adaptive')
r = jdca.gaussdca('alignment.a3m', window=100)
r = jdca.humphreys_dca('alignment.a3m')
r = jdca.ridge_dca('alignment.a3m')
```

## Limits

* Memory. A deep alignment, M + 1 ≥ D/√2, takes the Cholesky route (D = 20L for GaussDCA; for `ridge_dca` and `humphreys_dca`
  D = 21 × L rounded up to its size class). It holds three D × D float32 matrices on the GPU at its peak, whatever M (the
  sequences stream through in 256-row chunks): for GaussDCA 8.0 GB at L = 1,280 and 11.4 GB at L = 1,536 by XLA's memory
  analysis. JAX takes at most 75% of a GPU's memory by default (`XLA_PYTHON_CLIENT_MEM_FRACTION`), so on a 16 GB GPU
  the deep route fits up to about L = 1,500 in a fresh process. A long-running process that takes GPU memory as it goes
  (`XLA_PYTHON_CLIENT_PREALLOCATE=false`, as `jdca-server` does) can find its memory pool fragmented by earlier large maps, and a map
  near the limit then goes to the CPU; restarting the server clears that. A shallower alignment takes the Woodbury route, which holds
  M × D and M × M matrices instead. A GaussDCA map that does not fit in GPU memory is computed on the CPU (with a warning; the
  CPU route also holds the M × D rows in RAM), or use the windowed map.
* XLA's CPU thread pool ignores `OMP_NUM_THREADS`. Use `taskset` to bound the CPU cores the identity counts use.

## Tests

```
pip install -e ".[test]"
PYTHONPATH=.:tests pytest tests
```

The tests compare jdca with float64 transcriptions of GaussDCA.jl's estimator, of the ridge DCA and, step by step, of the TensorFlow
graph of Humphreys et al. (`tests/reference.py`), on the CPU and, when JAX sees one, the GPU.

## Acknowledgements

jdca rests on other people's work. GaussDCA, and the clear, carefully written GaussDCA.jl, are by Carlo Baldassi and colleagues;
jdca's defaults were checked against GaussDCA.jl's code and output, and its CPU identity kernel follows DCAUtils.jl. The ridge DCA
and `humphreys_dca` follow the GPU DCA of Ian R. Humphreys, Jimin Pei and colleagues; we thank Jing Zhang, Ian R. Humphreys,
Jimin Pei and their co-authors for publishing their script, which made an exact comparison possible. APC is due to Dunn, Wahl and
Gloor (2008).

## Citing

* GaussDCA (`jdca.gaussdca`, all modes): Baldassi et al. 2014.
* `jdca.humphreys_dca` and `jdca.ridge_dca`: Humphreys et al. 2021 and Zhang et al. 2025.

## References

* C. Baldassi, M. Zamparo, C. Feinauer, A. Procaccini, R. Zecchina, M. Weigt, A. Pagnani. Fast and accurate multivariate Gaussian
  modeling of protein families: predicting residue contacts and protein-interaction partners. *PLoS ONE* 9, e92721 (2014).
* I. R. Humphreys, J. Pei, M. Baek, et al. Computed structures of core eukaryotic protein complexes. *Science* 374, eabm4805 (2021).
* J. Zhang, I. R. Humphreys, J. Pei, et al. Predicting protein-protein interactions in the human proteome. *Science* (2025),
  doi:10.1126/science.adt1630. Preprint: Computing the human interactome, bioRxiv 2024.10.01.615885 (2024); the DCA script is in
  its supplementary Methods M5.3.
* S. D. Dunn, L. M. Wahl, G. B. Gloor. Mutual information without the influence of phylogeny or entropy dramatically improves
  residue contact prediction. *Bioinformatics* 24, 333–340 (2008).
