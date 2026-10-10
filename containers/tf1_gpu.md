# tf1_gpu.sif — TensorFlow 1.15 on Ampere (RTX A4000, sm_86)

Purpose: run the TF1 DCA script of Zhang, Humphreys, Pei et al. (Science 2025; bioRxiv 2024.10.01.615885, supplementary
Methods M5.3) on the GPU, as published, for the comparison in docs/comparison.md. Stock TF 1.15 wheels are CUDA 10 and have no Ampere support, so this is NVIDIA's NGC TF1 build.

## Image

| | |
|---|---|
| source | `docker://nvcr.io/nvidia/tensorflow:23.03-tf1-py3` (the newest `*-tf1-py3` tag on nvcr.io; NVIDIA stopped TF1 releases after 23.03) |
| manifest-list digest | `sha256:9882254598d633bb2d6104787b5ea0bcb47b97aa56934192c28dc98ff7f6386a` |
| linux/amd64 manifest digest | `sha256:9ef17803b84c172d0815084b7cc91736f3489abefa89001e3dccc3368c9ea7b4` (config `sha256:a23820905389c946dbf5514a1b8bbbf5e5a5b8f2a20440040380034b023c6fa1`; 49 layers, 8.15 GB compressed) |
| NGC build | `com.nvidia.build.id 54557773`, ref `1c6eff2a8529d7a7da1f7dc3385991e3283a64c3` |
| TensorFlow | 1.15.5 (`NVIDIA_TENSORFLOW_VERSION=23.03-tf1`, i.e. 1.15.5+nv23.03), Python 3.8.10, numpy 1.22.2 |
| CUDA | 12.1 (`CUDA_VERSION=12.1.0.023`), cuBLAS 12.1.0.26, cuSOLVER 11.4.4.55, cuDNN 8.8.1.3+cuda12.0, NCCL 2.17.1 |
| SIF | `tf1_gpu.sif`, 8,119,410,688 bytes (7.6 GiB), sha256 `af60f566f0d08faf55dd937845c86dd6fbb14ffe0fe1319fbb25f6c6d969017d` |
| built | 2026-10-08, apptainer 1.5.4 (unprivileged), host driver 580.178.04 (CUDA 13.0) |

Build command (pulled anonymously; cache/tmp outside /tmp, 8 cores):

```bash
B=$PWD/containers/.build_tf1; mkdir -p $B/tmp $B/cache
APPTAINER_TMPDIR=$B/tmp APPTAINER_CACHEDIR=$B/cache GOMAXPROCS=8 nice -n 10 \
  apptainer build --mksquashfs-args "-processors 8" tf1_gpu.sif docker://nvcr.io/nvidia/tensorflow@sha256:9882254598d633bb2d6104787b5ea0bcb47b97aa56934192c28dc98ff7f6386a
APPTAINER_CACHEDIR=$B/cache apptainer cache clean -f; rm -rf $B
```

Peak scratch use was about 23 GB (7.6 GB layer cache + 15 GB unpacked rootfs) plus the 8.1 GB SIF.

## Driver notes

- Use `apptainer exec --nv`. The image's env hook (`/etc/shinit_v2`) prints
  `_CUDA_COMPAT_STATUS=... CUDA_ERROR_SYSTEM_DRIVER_MISMATCH cuInit()=803` on this host. This is **benign**: the host driver
  (580, CUDA 13.0) is newer than the image's forward-compat libcuda (530.30.02). `/usr/local/cuda/compat/lib` does not
  exist in the SIF (only `lib.real`), so the loader uses the host libcuda bound by `--nv` at `/.singularity.d/libs`.
  Verified: TF creates `/device:GPU:0 ... compute capability: 8.6`, and MatrixInverse / MatMul run on GPU:0.
- **TF32**: NGC TF1 enables TF32 tensor-core matmuls on Ampere by default. Stock TF 1.15 has no TF32, so the
  benchmark runner sets `NVIDIA_TF32_OVERRIDE=0` (strict FP32) by default. On a 289 x 341 test alignment, GPU-vs-CPU
  differences in the score matrix were: FP32 median relative 5.9e-7, max relative 1.35e-3; TF32 median 6.1e-5,
  max 8.3e-3. Top-L pairs were identical in both modes.
- The harness sets `gpu_options.allow_growth=True` (TF1 otherwise reserves almost all free GPU memory on a shared card).

## Usage

`bench/humphreys/run_tf1.sh IN.npz OUT.npz`. See that script's header for the env knobs
(`NVIDIA_TF32_OVERRIDE`, `TF32_DEFAULT`, `REPEAT`, `PLACEMENT_CHECK`, `TF1_SIF`).
