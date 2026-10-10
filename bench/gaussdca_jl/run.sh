#!/usr/bin/env bash
# Run the ORIGINAL GaussDCA.jl (frozen in containers/gaussdca_jl.sif) on one alignment.
#
#   run.sh <alignment> <out_ranking.txt> [threads=1] [pseudocount=0.8] [theta=auto] [max_gap_fraction=0.9]
#
# threads sets JULIA_NUM_THREADS AND OPENBLAS_NUM_THREADS (gDCA's inv(cholesky(C)) is
# BLAS work; left unset, OpenBLAS would take its own default thread count regardless of
# the budget). The environment is cleaned (--cleanenv), so host JULIA_* settings do not leak in.
#
# a3m input: GaussDCA.jl drops lowercase letters and '.', but requires every record to have the
# same RAW length (a2m-style padding). An a3m with insertions fails with "inputs are not
# aligned". If the input has lowercase residues, this wrapper deletes them (a3m -> aligned
# FASTA) into <out_ranking>.input.fasta and runs on that file (kept, so a re-implementation
# can be run on the identical input). The conversion is not part of the reported timings.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
    sed -n '2,6p' "$0" >&2
    exit 2
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sif="${GAUSSDCA_SIF:-$(cd "$here/../.." && pwd)/containers/gaussdca_jl.sif}"
aln="$(realpath "$1")"
out="$(realpath -m "$2")"
threads="${3:-1}"
shift $(( $# >= 3 ? 3 : 2 ))

[[ -f "$aln" ]] || { echo "run.sh: no such alignment: $aln" >&2; exit 1; }
[[ -f "$sif" ]] || { echo "run.sh: no image at $sif (set GAUSSDCA_SIF)" >&2; exit 1; }
mkdir -p "$(dirname "$out")"

if awk '!/^>/ && /[a-z.]/ {f = 1; exit} END {exit !f}' "$aln"; then
    conv="${out}.input.fasta"
    # header lines unchanged; sequence lines: delete insertions (lowercase) and '.'
    awk '/^>/ {print; next} {gsub(/[a-z.]/, ""); print}' "$aln" > "$conv"
    echo "run.sh: lowercase insertions found; a3m -> aligned FASTA: $conv" >&2
    aln="$conv"
fi

binds=()
for d in "$(dirname "$aln")" "$(dirname "$out")" "$here"; do
    case "$d/" in "$HOME"/*) ;; *) binds+=(--bind "$d") ;; esac   # $HOME is mounted by default
done

exec apptainer exec --cleanenv \
    --env JULIA_NUM_THREADS="$threads" \
    --env OPENBLAS_NUM_THREADS="$threads" \
    "${binds[@]}" \
    "$sif" julia --startup-file=no "$here/run_gdca.jl" "$aln" "$out" "$@"
