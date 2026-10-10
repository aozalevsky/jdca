# Benchmark driver for the ORIGINAL GaussDCA.jl (run inside gaussdca_jl.sif).
#
#   julia run_gdca.jl <alignment.fasta> <out_ranking.txt> [pseudocount=0.8] [theta=auto] [max_gap_fraction=0.9]
#
# Calls gDCA(file; pseudocount, θ, max_gap_fraction) twice in one process on the same file:
# the first (cold) call includes JIT compilation, the second (warm) call is the timed one.
# The ranking of the warm call is written with printrank ("i j score", 1-based, sorted by
# decreasing score, |i-j| >= 5 = GaussDCA's default min_separation, APC-corrected Frobenius
# norm = default score). gDCA's own progress lines (θ, threshold, M, N, Meff) go to stdout;
# one JSON line with timings and versions goes to stderr.
#
# Input: aligned FASTA. As in GaussDCA, lowercase letters and '.' are dropped, but every
# record must have the same raw length INCLUDING them (a2m-style). An a3m with insertions
# fails with "inputs are not aligned": delete the lowercase letters first (see run.sh).

const t_start = time()
using GaussDCA, DCAUtils, LinearAlgebra
import TOML
const t_loaded = time()

function usage()
    println(stderr, "usage: julia run_gdca.jl <alignment.fasta> <out_ranking.txt> ",
            "[pseudocount=0.8] [theta=auto] [max_gap_fraction=0.9]")
    exit(2)
end

(2 <= length(ARGS) <= 5) || usage()
const infile = ARGS[1]
const outfile = ARGS[2]
const pseudocount = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.8
const theta = (length(ARGS) >= 4 && ARGS[4] != "auto") ? parse(Float64, ARGS[4]) : :auto
const max_gap_fraction = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.9

# @timed plus the JIT time spent inside the call (Base.cumulative_compile_time_ns, Julia >= 1.6).
# invokelatest is a dynamic-dispatch barrier: without it, compiling run_once would infer and
# compile the whole gDCA call tree BEFORE the timer starts, hiding the JIT from the cold time.
function run_once()
    c0 = Base.cumulative_compile_time_ns()[1]
    r = @timed Base.invokelatest(gDCA, infile; pseudocount = pseudocount, θ = theta,
                                 max_gap_fraction = max_gap_fraction)
    flush(stdout)
    return r, (Base.cumulative_compile_time_ns()[1] - c0) / 1e9
end

Base.cumulative_compile_timing(true)
println("## cold call (includes JIT)"); flush(stdout)
cold, cold_jit = run_once()
t_gc = @elapsed GC.gc()
println("## warm call (timed)"); flush(stdout)
warm, warm_jit = run_once()
Base.cumulative_compile_timing(false)
R = warm.value
t0 = time()
printrank(outfile, R)
t_write = time() - t0

# Not timed: recompute what gDCA used internally (same functions, same arguments), quietly.
t0 = time()
Z = read_fasta_alignment(infile, max_gap_fraction)
N, M = size(Z)
θ_used = theta === :auto ? compute_theta(Z) : Float64(theta)
_, Meff = compute_weights(Z, θ_used; verbose = false)
t_stats = time() - t0

# Package versions/revisions as resolved in the image's Manifest.toml.
manifest = TOML.parsefile(joinpath(dirname(Base.active_project()), "Manifest.toml"))
function pkginfo(name)
    e = manifest["deps"][name][1]
    d = Dict{String,Any}("version" => get(e, "version", nothing),
                         "git_tree_sha1" => get(e, "git-tree-sha1", nothing))
    haskey(e, "repo-rev") && (d["git_rev"] = e["repo-rev"])
    return d
end

jsonval(x::Nothing) = "null"
jsonval(x::Bool) = x ? "true" : "false"
jsonval(x::Integer) = string(x)
jsonval(x::AbstractFloat) = isfinite(x) ? string(x) : "null"
jsonval(x::AbstractString) = "\"" * escape_string(x) * "\""
jsonval(x::Symbol) = jsonval(String(x))
jsonval(x::VersionNumber) = jsonval(string(x))
jsonval(d::AbstractDict) = "{" * join(("$(jsonval(string(k))): $(jsonval(v))" for (k, v) in sort(collect(d); by = first)), ", ") * "}"

info = Dict{String,Any}(
    "tool" => "GaussDCA.jl",
    "input" => abspath(infile),
    "output" => abspath(outfile),
    "pseudocount" => pseudocount,
    "theta_arg" => theta === :auto ? "auto" : theta,
    "theta" => θ_used,
    "max_gap_fraction" => max_gap_fraction,
    "N" => N,                       # alignment columns kept (upper-case + '-')
    "M" => M,                       # sequences passing max_gap_fraction
    "Meff" => Meff,
    "n_pairs" => length(R),
    "t_gdca_warm_s" => warm.time,
    "t_gdca_cold_s" => cold.time,
    "warm_alloc_bytes" => warm.bytes,
    "warm_gc_s" => warm.gctime,
    "cold_jit_s" => cold_jit,      # compile time inside the cold call (summed over threads)
    "warm_jit_s" => warm_jit,      # should be ~0
    "t_write_s" => t_write,
    "t_gc_s" => t_gc,              # full GC between the two calls
    "t_stats_s" => t_stats,        # the untimed θ/Meff recomputation above
    "t_load_s" => t_loaded - t_start,
    "t_total_s" => time() - t_start,
    "identical_runs" => cold.value == warm.value,
    "nthreads" => Threads.nthreads(),
    "blas_threads" => BLAS.get_num_threads(),
    "blas" => string(BLAS.get_config().loaded_libs[1].libname |> basename),
    "julia" => string(VERSION),
    "GaussDCA" => pkginfo("GaussDCA"),
    "DCAUtils" => pkginfo("DCAUtils"),
    "FastaIO" => pkginfo("FastaIO"),
    "hostname" => gethostname(),
    "cpu" => Sys.cpu_info()[1].model,
)
println(stderr, jsonval(info))
