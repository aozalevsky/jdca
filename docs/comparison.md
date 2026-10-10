# Comparison: GaussDCA.jl, jdca, and the DCA of Humphreys et al.

jdca builds on two published DCA implementations. This note runs both as their authors published them, next to jdca:

* **GaussDCA.jl**, the reference implementation of GaussDCA by Carlo Baldassi and colleagues (Baldassi et al., *PLoS ONE* 2014).
* **The GPU DCA used by Humphreys et al.** (*Science* 2021), as the TensorFlow 1 script that Jing Zhang, Ian R. Humphreys, Jimin Pei
  and colleagues published with their human interactome work (Zhang et al., *Science* 2025; the script and its APC formula are in the
  preprint's supplementary Methods M5.3). There it ranks protein pairs by their coevolution signal before structure prediction.

We are grateful to both groups for making their code available; an exact comparison would not be possible otherwise. Each was
written for its own purpose: GaussDCA.jl as an exact float64 reference, the TensorFlow script as a compact step in a proteome-wide
pipeline. The timings below compare implementations on one workstation, not the merits of the methods.

jdca runs three estimators against them:

* `jdca.gaussdca`, GaussDCA.jl's estimator ([benchmark.md](benchmark.md)), and its `adaptive` mode ([adaptive.md](adaptive.md));
* `jdca.humphreys_dca`, the estimator of Humphreys et al. with the settings of their script;
* `jdca.ridge_dca`, a variant of it with a per-column ridge, scored by partial correlations.

The targets, inputs and hardware are those of [benchmark.md](benchmark.md): three AlphaFold DB v6 entries, an RTX A4000 GPU, and
8 CPU cores for GaussDCA.jl.

## The estimators

| | GaussDCA (GaussDCA.jl, `jdca.gaussdca`) | Humphreys et al. (their script, `jdca.humphreys_dca`) | `jdca.ridge_dca` |
|---|---|---|---|
| states | 20 amino acids (the gap is the reference state) | 21 (the gap is a state) | 21 |
| sequence filter | > 90% gaps removed | none | none |
| reweighting | Hamming distance < ⌊θL⌋, θ = auto | identity > 0.8 L | identity > 0.8 L |
| regularisation | pseudocount 0.8 (mixing with the uniform distribution) | one ridge 4.5 / √Meff on the covariance | ridge 4.07 / √nᵢ per column (nᵢ = weighted non-gap count) |
| inverse | GaussDCA.jl: float64 Cholesky; jdca: Woodbury or Cholesky in float32 | their script: float32 `tf.linalg.inv` of the full (21 L)² matrix; jdca: Woodbury or Cholesky in float32 | Woodbury or Cholesky in float32 |
| pair score | Frobenius norm of the zero-sum-gauged 20 × 20 block | Frobenius norm of the raw 20 × 20 non-gap block | Frobenius norm of the 20 × 20 block of partial correlations |
| correction | APC, S − SᵢSⱼ / (S.. (1 − 1/L)) | APC, S − SᵢSⱼ / S.. | as GaussDCA |

## How their script was run

* **The code.** Their script from the supplement, with the model code unchanged. Only the input/output and timing harness around
  it is ours (`bench/humphreys/humphreys_dca_tf1.py`).
* **The runtime.** The TensorFlow 1.15 wheels predate current GPUs, so it runs in NVIDIA's NGC TensorFlow 1.15.5 container
  (`nvcr.io/nvidia/tensorflow:23.03-tf1-py3`, CUDA 12.1; `containers/tf1_gpu.md`). TF32 tensor-core arithmetic is switched off
  (`NVIDIA_TF32_OVERRIDE=0`), as in stock TF 1.15.
* **Checks.** Every key operation (one-hot, matrix products, the inverse) ran on the GPU, recorded once per target. On a test
  alignment, its GPU output and stock TF 1.15.5 on the CPU agree to 1.4e-3 relative at worst and share their top-L pairs
  (`containers/tf1_gpu.md`).
* **Input and scoring.** As in their script, every sequence of the alignment is kept. (In their pipeline the paired alignments were
  first filtered to 90% identity with HHfilter; here all methods read the same unfiltered alignment.) Their APC formula, for a
  single protein, is applied to the raw scores their script returns.

## `jdca.humphreys_dca` reproduces their script

| | P34884 | P0A9S1 | P57740 |
|---|---|---|---|
| max \|ΔS\| / max \|S\| (their raw scores) | 8.9e-5 | 4.7e-4 | 9.4e-5 |
| top-L pairs shared (after their APC) | 100% | 100% | 100% |

The differences are float32 rounding. The largest is on the deep alignment, P0A9S1 with 17,508 sequences; our guess, not checked
against float64, is the covariance, which their script forms as one float32 GPU contraction over all rows and jdca in 256-row chunks.

## Contact precision

Precision against the AFDB model: Cβ–Cβ < 8 Å, counting only pairs whose two residues have pLDDT ≥ 70. Top L/5, L/2 and L pairs:

| | method | \|i−j\| ≥ 6: L/5 | L/2 | L | \|i−j\| ≥ 24: L/5 | L/2 | L |
|---|---|---|---|---|---|---|---|
| **P34884** | GaussDCA.jl = jdca GaussDCA | 0.57 | 0.49 | 0.42 | 0.61 | 0.47 | 0.41 |
| L 115, M 2,548 | jdca GaussDCA, adaptive mode | 0.70 | 0.47 | 0.38 | 0.61 | 0.47 | 0.41 |
| | Humphreys et al. (their script) = `jdca.humphreys_dca` | 0.70 | 0.47 | 0.42 | 0.61 | 0.49 | 0.40 |
| | jdca ridge | 0.65 | 0.53 | 0.39 | 0.57 | 0.46 | 0.38 |
| **P0A9S1** | GaussDCA.jl = jdca GaussDCA | 0.89 | 0.81 | 0.71 | 0.91 | 0.78 | 0.69 |
| L 382, M 17,508 | jdca GaussDCA, adaptive mode | 0.89 | 0.76 | 0.68 | 0.88 | 0.79 | 0.69 |
| | Humphreys et al. (their script) = `jdca.humphreys_dca` | 0.89 | 0.78 | 0.69 | 0.88 | 0.78 | 0.69 |
| | jdca ridge | 0.91 | 0.81 | 0.72 | 0.89 | 0.79 | 0.73 |
| **P57740** | GaussDCA.jl = jdca GaussDCA | 0.35 | 0.21 | 0.14 | 0.22 | 0.12 | 0.08 |
| L 925, M 1,883 | jdca GaussDCA, adaptive mode | 0.32 | 0.19 | 0.15 | 0.19 | 0.13 | 0.10 |
| | Humphreys et al. (their script) = `jdca.humphreys_dca` | 0.32 | 0.19 | 0.13 | 0.19 | 0.11 | 0.09 |
| | jdca ridge | 0.35 | 0.21 | 0.14 | 0.18 | 0.10 | 0.09 |

Their script and `jdca.humphreys_dca` have the same precision on every entry. M is the number of sequences each method reads: all of
them, except GaussDCA's 90% gap filter (P0A9S1 17,507, P57740 1,647).

Three targets cannot rank the estimators: the differences change sign from target to target and are a few contacts each. Telling
them apart needs a larger set of targets.

## Other contact metrics

Top-k precision is one view of a contact map. On seven AlphaFold DB entries (the three above and the four longer chains of
[windowed.md](windowed.md); L 115–1,203), the four estimators were also scored with the following, at the top L/5, L/2 and L pairs
(same truth as above, pairs with |i − j| ≥ 6):

* *long-range*: precision among the top pairs with |i − j| ≥ 24;
* *tolerant*: the fraction of pairs with a true contact within ±1 in i and in j;
* *patch recall*: true contacts within ±2 of each other in i and in j form patches; the fraction of patches hit;
* *SSE-pair*: helix and strand segments of ≥ 3 residues (P-SEA, from biotite); the fraction of segment pairs in contact that are hit;
* *redundancy*: correct predictions per hit patch (lower means the correct predictions are spread over more patches);
* *average precision* over the whole ranking.

The GaussDCA column is `jdca.gaussdca` (GaussDCA.jl's estimator), and the Humphreys et al. column is `jdca.humphreys_dca`, which
reproduces their script (above). Means over the seven targets, with the best value of each row in bold (for redundancy, the
lowest; values equal at the three decimals shown are all bold):

| metric | top | GaussDCA | adaptive | Humphreys et al. | ridge |
|---|---|---|---|---|---|
| precision | L/5 | 0.553 | 0.566 | 0.562 | **0.568** |
| precision | L/2 | **0.434** | 0.423 | 0.415 | 0.431 |
| precision | L | **0.340** | 0.331 | 0.329 | 0.335 |
| long-range | L/5 | **0.520** | **0.520** | 0.513 | 0.504 |
| long-range | L/2 | 0.381 | **0.390** | 0.383 | 0.377 |
| long-range | L | 0.294 | **0.300** | 0.296 | 0.298 |
| tolerant | L/5 | 0.658 | 0.662 | 0.661 | **0.664** |
| tolerant | L/2 | **0.552** | 0.541 | 0.529 | 0.547 |
| tolerant | L | 0.462 | 0.458 | 0.455 | **0.465** |
| patch recall | L/5 | 0.155 | **0.166** | 0.164 | 0.161 |
| patch recall | L/2 | **0.270** | 0.261 | 0.265 | **0.270** |
| patch recall | L | **0.352** | 0.346 | 0.351 | 0.351 |
| SSE-pair | L/5 | 0.340 | 0.342 | 0.346 | **0.349** |
| SSE-pair | L/2 | 0.495 | **0.511** | 0.504 | 0.498 |
| SSE-pair | L | **0.591** | 0.577 | 0.570 | 0.581 |
| redundancy | L/5 | 1.399 | **1.361** | 1.370 | 1.403 |
| redundancy | L/2 | 1.578 | 1.578 | **1.530** | 1.593 |
| redundancy | L | 1.863 | 1.856 | **1.816** | 1.859 |
| average precision | | 0.181 | 0.181 | 0.179 | **0.184** |

**There is no clear winner.**
* The four means of every precision, recall and coverage row lie within 0.023 of each other (the redundancy rows within 0.063),
  and the best estimator changes from row to row.
* On single targets the order keeps changing. Of the 112 target × metric cases (redundancy excluded), each estimator is the best
  (ties counted for each) in: ridge 50, adaptive 37, GaussDCA 32, Humphreys et al. 18.
* Seven targets are far too few to separate differences of this size, so the bold marks the best mean here, not a better method.

## Time

Each program timed three ways, as in [benchmark.md](benchmark.md):

* *one-off*: one command from process start to exit;
* *one-off with jdca-server*: the same jdca command while `jdca-server` runs;
* *batch*: one map in a process that has already computed one. For their script that is a later `sess.run` in the same session,
  which starts from the encoded alignment, and the jdca columns here are timed the same way (from the encoded alignment to the
  scores). GaussDCA.jl's batch time, from [benchmark.md](benchmark.md), also includes reading the file and building the ranking.

| | | GaussDCA.jl, 8 threads (CPU) | Humphreys et al., TF1 (GPU) | jdca GaussDCA (GPU) | jdca adaptive (GPU) | `jdca.humphreys_dca` (GPU) | jdca ridge (GPU) |
|---|---|---|---|---|---|---|---|
| **P34884** (L 115) | one-off | 9.9 s | 4.30 s | 1.73 s | 1.65 s | 1.70 s | 1.72 s |
| | one-off with jdca-server | | | 0.19 s | 0.19 s | 0.19 s | 0.19 s |
| | batch, per map | 0.37 s | 0.023 s | 0.023 s | 0.020 s | 0.021 s | 0.021 s |
| **P0A9S1** (L 382) | one-off | 16.3 s | 5.06 s | 2.43 s | 2.20 s | 2.35 s | 2.35 s |
| | one-off with jdca-server | | | 0.87 s | 0.78 s | 0.86 s | 0.86 s |
| | batch, per map | 6.47 s | 0.79 s | 0.48 s | 0.43 s | 0.51 s | 0.51 s |
| **P57740** (L 925) | one-off | 57.7 s | 6.56 s | 2.38 s | 2.34 s | 2.24 s | 2.27 s |
| | one-off with jdca-server | | | 0.91 s | 0.92 s | 0.92 s | 0.92 s |
| | batch, per map | 47.1 s | 2.27 s | 0.18 s | 0.19 s | 0.20 s | 0.20 s |

Each time is the fastest of three runs. The GaussDCA.jl column is from [benchmark.md](benchmark.md) (its one-off time without our
driver's second call). The other columns were measured one target at a time on the shared machine, and a target's run was kept only
if no other program used the GPU while it ran (otherwise it was repeated); the rest of the machine was nearly idle, at about 0.1 CPU
of other programs' work on average. Their script and jdca were timed in separate guarded runs, one after the other; `bench/guard.py`'s verdicts are in `compare.json` (`guard_humphreys` and `guard`). As in
[benchmark.md](benchmark.md), NumPy's transparent huge pages were switched off for every timed program, inside the container too.

* **Batch.** On the short chain their script and jdca take the same time per map. On the deep alignment `jdca.humphreys_dca`
  takes 0.51 s against 0.79 s; we have not profiled where the difference comes from (likely candidates are the Cholesky inverse of
  the symmetric matrix and the int8 identity counts). On the long chain with a shallow
  alignment, their script inverts the full (21 L)² = 19,425² covariance, a simple and general formulation; jdca's Woodbury route
  factorises an M × M matrix instead (0.20 s against 2.27 s per map).
* **One-off.** Their one-off command includes starting the container (0.27 s), importing TensorFlow (2.1–2.3 s) and starting its
  session (0.7 s); jdca's includes about 1.5 s of GPU start-up, which `jdca-server` removes. Their pipeline runs the script in
  batch, with thousands of alignments per file in one session, so the batch rows are the ones that reflect how it is used.
* **jdca's settings.** All four take about the same time. The adaptive mode is a little quicker on the deep alignment: its
  reweighting needs no all-pairs mean-identity pass.

## Reproduce

```
apptainer build containers/gaussdca_jl.sif containers/gaussdca_jl.def
apptainer build containers/tf1_gpu.sif \
  docker://nvcr.io/nvidia/tensorflow@sha256:9882254598d633bb2d6104787b5ea0bcb47b97aa56934192c28dc98ff7f6386a   # 23.03-tf1-py3
python bench/benchmark.py P34884 P0A9S1 P57740     # GaussDCA.jl vs jdca; writes the aligned inputs
python bench/compare.py P34884 P0A9S1 P57740       # their script, then jdca's other estimators
python bench/windowed.py P0A9S1 P57740 P49588 P53396 P29474 P17301   # writes the inputs of the four longer chains
python bench/metrics.py P34884 P0A9S1 P57740 P49588 P53396 P29474 P17301
```

The numbers are in `bench/out/results.json`, `bench/out/compare.json` and `bench/out/metrics.json`. For timings to compare, run
each target on its own under `bench/guard.py`, as in [benchmark.md](benchmark.md), e.g.
`python bench/guard.py --json bench/out/compare.json --key P34884 -- python bench/compare.py P34884`.
