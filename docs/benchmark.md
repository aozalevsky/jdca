# Benchmark: jdca vs GaussDCA.jl

jdca's GaussDCA against GaussDCA.jl, the reference implementation by Carlo Baldassi and colleagues (Baldassi et al., *PLoS ONE*
2014), on three AlphaFold DB entries: a short chain with a normal-sized alignment, a medium chain with a deep alignment, and a long
chain with a normal-sized alignment. Both programs read the same alignment with the same settings (GaussDCA.jl's defaults).

## Setup

**Targets.** AlphaFold DB v6 models and the MSAs AFDB distributes with them (`AF-<id>-F1-model_v6.cif`, `AF-<id>-F1-msa_v6.a3m`):

| UniProt | protein | L | sequences in the MSA | kept by the 90% gap filter | Meff | θ (auto) | residues with pLDDT ≥ 70 |
|---|---|---|---|---|---|---|---|
| P34884 | macrophage migration inhibitory factor (mouse) | 115 | 2,548 | 2,548 | 620.5 | 0.4168 | 100% |
| P0A9S1 | lactaldehyde reductase (*E. coli*) | 382 | 17,508 | 17,507 | 2,546.1 | 0.3652 | 100% |
| P57740 | nuclear pore complex protein Nup107 (human) | 925 | 1,883 | 1,647 | 322.4 | 0.2720 | 77% |

**Input.** GaussDCA.jl reads aligned FASTA, in which every record has the same length, so the A3M insertions (lowercase letters
and `.`) were removed first. Both programs read that aligned FASTA.

**Software.**
* GaussDCA.jl 2.0.0 (commit `067016f`) with DCAUtils 1.1.0 and FastaIO 1.1.0, on Julia 1.10.12 with OpenBLAS 0.3.23. It runs
  from a pinned Apptainer image (`containers/gaussdca_jl.def`), with `JULIA_NUM_THREADS` and `OPENBLAS_NUM_THREADS` both set to the
  thread count.
* jdca 0.1.0 on JAX 0.9.2, NumPy 2.4.3 and SciPy 1.17.1 (OpenBLAS).

**Hardware.** 2 × Intel Xeon Silver 4214R (24 cores) and one NVIDIA RTX A4000 (16 GB, driver 580). For each target, GaussDCA.jl
and jdca on the CPU were pinned with `taskset` to the same 8 physical cores of one socket, one hyper-thread each.

**Timing.** Three numbers per program, for three ways of using it (see the README, "How long a map takes"):
* *One-off*: one command, from process start to exit. For GaussDCA.jl that includes Julia's start-up and the compilation of its
  first call; for jdca, Python's start-up and (on the GPU) GPU initialisation. jdca's GPU programs were already in its persistent
  cache (the first alignment of a new size class adds 1–2 s of one-time compilation); its CPU programs are compiled in every
  process, and that is part of the CPU one-off time.
* *One-off with jdca-server*: the same jdca command while `jdca-server` runs.
* *Batch*: one map in a process that has already computed one, from reading the alignment file to the ranking: a second
  `gDCA(file)` call in the same Julia session, or a repeated `jdca.ranking(jdca.gaussdca(file).scores)` in the same Python
  process.

## Same results

| | θ, Meff | max \|ΔS\| / max \|S\| | top-L pairs shared | contact precision |
|---|---|---|---|---|
| P34884 | identical | 1.5e-5 | 100% | identical |
| P0A9S1 | identical | 7.1e-5 | 100% | identical |
| P57740 | identical | 8.7e-7 | 100% | identical |

S is the APC-corrected score matrix and ΔS the difference between the two programs over all ranked pairs (|i − j| ≥ 5). Both
programs build the reweighting from exact integer identity counts, so θ and Meff agree to the last printed digit. The remaining
score difference is jdca's float32 arithmetic (GaussDCA.jl works in float64); it changes no top-L pair.

Contact precision is the fraction of true contacts among the top-ranked pairs. A contact is a Cβ–Cβ distance below 8 Å (Cα for
glycine) in the AFDB model, counting only pairs where both residues have pLDDT ≥ 70. It is the same for both programs on every
metric:

| | \|i−j\| ≥ 6: top L/5 | L/2 | L | \|i−j\| ≥ 24: top L/5 | L/2 | L |
|---|---|---|---|---|---|---|
| P34884 | 0.57 | 0.49 | 0.42 | 0.61 | 0.47 | 0.41 |
| P0A9S1 | 0.89 | 0.81 | 0.71 | 0.91 | 0.78 | 0.69 |
| P57740 | 0.35 | 0.21 | 0.14 | 0.22 | 0.12 | 0.08 |

## Time

| | | GaussDCA.jl, 1 thread | GaussDCA.jl, 8 threads | jdca, GPU | jdca, CPU (8 cores) |
|---|---|---|---|---|---|
| **P34884** (L 115, M 2,548) | one-off | 11.4 s | 9.9 s | 1.71 s | 1.50 s |
| | one-off with jdca-server | | | 0.19 s | |
| | batch, per map | 0.84 s | 0.37 s | **0.036 s** | 0.29 s |
| **P0A9S1** (L 382, M 17,507) | one-off | 37.1 s | 16.3 s | 2.35 s | 9.17 s |
| | one-off with jdca-server | | | 0.87 s | |
| | batch, per map | 26.1 s | 6.47 s | **0.59 s** | 7.77 s |
| **P57740** (L 925, M 1,647) | one-off | 222.4 s | 57.7 s | 2.39 s | 7.89 s |
| | one-off with jdca-server | | | 0.92 s | |
| | batch, per map | 211.7 s | 47.1 s | **0.28 s** | 6.09 s |

jdca's times are the fastest of three runs. GaussDCA.jl's batch time is its second call, timed inside Julia. Its one-off time is the
measured process minus what our driver adds after the first call (a garbage collection, the timed second call, and a recomputation
of θ and Meff for this report): Julia's start-up, package loading, the first `gDCA` call with its compilation (7.2–9.8 s by
Julia's own count) and writing the ranking.

The machine is shared with other users, so each target was timed on its own and its run was kept only if, while it ran, no other
program used the GPU and other programs' use of the 8 pinned cores (either hyper-thread) stayed below 30% of one CPU in every
10-second window (mostly the kernel's I/O threads); otherwise it was repeated. `bench/guard.py` checks these conditions and stores
its verdict with each target's record (`guard` in `results.json`). NumPy's requests for transparent huge pages were
switched off for every timed program (`NUMPY_MADVISE_HUGEPAGE=0`): on this machine's fragmented memory they could stall a large
allocation for 0.1 s or more.

* **Batch.** Per map, jdca on the GPU is 10–170× faster than GaussDCA.jl on 8 threads.
* **One-off.** Start-up dominates both: GaussDCA.jl's first call compiles its code, and jdca spends about 1.5 s starting the GPU.
  With `jdca-server` running, a jdca command takes 0.2–0.9 s.
* **Long chain, shallow alignment.** GaussDCA.jl inverts the dense (20 L)² = 18,500² covariance in float64, the natural choice for
  a reference implementation. jdca's Woodbury route only factorises an M × M matrix, so it is faster on the CPU too (6.1 s against
  47.1 s per map at 8 threads).
* **Deep alignment on the CPU.** Here the two are close (7.8 s against GaussDCA.jl's 6.5 s per map). On the CPU jdca counts
  sequence identities on bit-packed sequences, as GaussDCA.jl does (on a CPU that is twice as fast as int8 products). It forms the
  pair statistics as a dense matrix product (BLAS `ssyrk`), while GaussDCA.jl's compiled loops visit only the residue pairs that
  occur, which pays off on a deep alignment.

## Beyond GaussDCA.jl

jdca's add-ons have their own notes: the adaptive mode ([adaptive.md](adaptive.md)), the windowed map for long chains
([windowed.md](windowed.md)), and the DCA of Humphreys et al. with the ridge DCA ([comparison.md](comparison.md)).

## Reproduce

```
apptainer build containers/gaussdca_jl.sif containers/gaussdca_jl.def
# bench/data/<UniProt>/ : the AlphaFold DB model and MSA of each target,
#   https://alphafold.ebi.ac.uk/files/AF-<id>-F1-model_v6.cif
#   https://alphafold.ebi.ac.uk/files/msa/AF-<id>-F1-msa_v6.a3m
for u in P34884 P0A9S1 P57740; do       # one target at a time; repeat one the guard reports as not clean (exit status 3)
  python bench/guard.py --cores --json bench/out/results.json --key $u -- \
    python bench/benchmark.py $u --threads 1,8 --cores {cores}
done
```

`bench/benchmark.py` writes the rankings and timings to `bench/out/` and all numbers to `bench/out/results.json`.
