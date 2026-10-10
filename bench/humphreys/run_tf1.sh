#!/usr/bin/env bash
# run_tf1.sh IN.npz OUT.npz
# Runs humphreys_dca_tf1.py (the TF1 DCA script of Zhang, Humphreys, Pei et al., Science 2025; bioRxiv 2024.10.01.615885 M5.3)
# on the GPU inside
# tf1_gpu.sif (NGC nvcr.io/nvidia/tensorflow:23.03-tf1-py3, TF 1.15.5+nv23.03, CUDA 12.1).
#
# Env passed through to the container:
#   NVIDIA_TF32_OVERRIDE  default 0 = strict FP32 matmuls (what stock TF 1.15 computes; NGC TF1 enables TF32 on Ampere
#                         otherwise). TF32_DEFAULT=1 leaves the variable unset = the NGC default (TF32 on): on a 289 x 341
#                         test alignment the GPU-vs-CPU median relative difference is then 6e-5 instead of 6e-7 (top-L
#                         unchanged; containers/tf1_gpu.md).
#   REPEAT                runs per key (default 2: first call vs a second call on the same input; 1 = once)
#   PLACEMENT_CHECK       1 (default) = one traced run of the first key to report op placement; 0 = skip
#   CUDA_VISIBLE_DEVICES  as usual
#   TF1_SIF               override the image path
set -euo pipefail
if [ $# -ne 2 ]; then echo "usage: $0 IN.npz OUT.npz" >&2; exit 2; fi
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIF=${TF1_SIF:-$(cd "$HERE/../.." && pwd)/containers/tf1_gpu.sif}
IN=$(realpath "$1")
OUT=$(realpath -m "$2")
mkdir -p "$(dirname "$OUT")"

BINDS=()
for d in "$HERE" "$(dirname "$IN")" "$(dirname "$OUT")"; do
    case "$d" in "$HOME"|"$HOME"/*) ;; *) BINDS+=(-B "$d") ;; esac   # $HOME is mounted by default
done

if [ "${TF32_DEFAULT:-0}" = "1" ]; then
    unset APPTAINERENV_NVIDIA_TF32_OVERRIDE
else
    export APPTAINERENV_NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-0}
fi
export APPTAINERENV_PYTHONNOUSERSITE=1            # never import ~/.local packages into the container's python
export APPTAINERENV_REPEAT=${REPEAT:-2}
export APPTAINERENV_PLACEMENT_CHECK=${PLACEMENT_CHECK:-1}
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && export APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export APPTAINERENV_PYTHONUNBUFFERED=1

exec apptainer exec --nv --cleanenv "${BINDS[@]}" "$SIF" python "$HERE/humphreys_dca_tf1.py" "$IN" "$OUT"
