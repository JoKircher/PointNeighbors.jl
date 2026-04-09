#
# Benchmark: effect of `estimate_max_points_per_cell` on memory and runtime
# for `FullGridCellList`, on CPU and (optionally) GPU.
#
# Usage:
#   julia --project=. benchmarks/estimate_max_points.jl
#
# The script builds a sparse 3D point cloud (dense cluster + sparse background),
# constructs `GridNeighborhoodSearch` with `FullGridCellList` using
#   (1) the default `max_points_per_cell = 100`
#   (2) a value returned by `estimate_max_points_per_cell`
# and measures:
#   - bytes allocated for the cell-list backing storage
#   - time for `initialize!`
#   - time for `update!`
#   - time for an n-body-style `foreach_point_neighbor` sweep
# on CPU, and on GPU if CUDA is available.

using PointNeighbors
using BenchmarkTools
using Random
using Printf

const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

function make_point_cloud(n_points; seed = 1)
    Random.seed!(seed)
    # Uniform distribution in [0, 1]^3 — the realistic case for SPH-like setups
    # where the default `max_points_per_cell = 100` heavily over-allocates.
    return rand(Float32, 3, n_points)
end

function build_nhs(coords, search_radius, max_points_per_cell)
    NDIMS = size(coords, 1)
    min_corner = ntuple(_ -> 0.0f0, NDIMS)
    max_corner = ntuple(_ -> 1.0f0, NDIMS)
    cell_list = FullGridCellList(; min_corner, max_corner,
                                 search_radius = Float32(search_radius),
                                 max_points_per_cell)
    return GridNeighborhoodSearch{NDIMS}(; search_radius = Float32(search_radius),
                                         n_points = size(coords, 2),
                                         cell_list)
end

# Bytes allocated for the cell-list backing storage (DynamicVectorOfVectors or Vector{Vector})
function cell_storage_bytes(nhs)
    cells = nhs.cell_list.cells
    if cells isa PointNeighbors.DynamicVectorOfVectors
        return sizeof(cells.backend) + sizeof(cells.lengths)
    elseif cells isa Vector{<:Vector}
        return sum(sizeof, cells; init = 0) + sizeof(cells)
    else
        return sizeof(cells)
    end
end

# n-body-style neighbor sweep that writes into a per-point accumulator
function nbody_sweep!(dv, coords, nhs, backend)
    dv .= 0
    foreach_point_neighbor(coords, coords, nhs;
                           parallelization_backend = backend) do i, _j, pos_diff, distance
        distance < sqrt(eps(Float32)) && return
        f = pos_diff / distance^3
        for d in axes(dv, 1)
            @inbounds dv[d, i] += f[d]
        end
    end
    return dv
end

function bench_one(label, coords_host, search_radius, mppc, backend; on_gpu::Bool)
    # Always build + initialize on CPU first (host-side init path is the only one that
    # works for FullGridCellList — GPU init goes through the same scalar code path).
    nhs_host = build_nhs(coords_host, search_radius, mppc)
    PointNeighbors.initialize!(nhs_host, coords_host, coords_host)

    if on_gpu
        coords = PointNeighbors.Adapt.adapt(backend, coords_host)
        nhs = PointNeighbors.Adapt.adapt(backend,
                                         PointNeighbors.freeze_neighborhood_search(nhs_host))
        # Time only update! and the n-body sweep on GPU.
        # (initialize! is host-only for FullGridCellList.)
        PointNeighbors.update!(nhs, coords, coords; parallelization_backend = backend)
        CUDA.synchronize()
        t_init = NaN
        t_update = @belapsed (PointNeighbors.update!($nhs, $coords, $coords;
                                                     parallelization_backend = $backend);
                              CUDA.synchronize()) samples=20 evals=1 seconds=10
        dv = similar(coords)
        nbody_sweep!(dv, coords, nhs, backend); CUDA.synchronize()
        t_sweep = @belapsed (nbody_sweep!($dv, $coords, $nhs, $backend);
                             CUDA.synchronize()) samples=20 evals=1 seconds=10
    else
        coords = coords_host
        nhs = nhs_host
        t_init = @belapsed begin
            local n = build_nhs($coords_host, $search_radius, $mppc)
            PointNeighbors.initialize!(n, $coords_host, $coords_host)
        end samples=20 evals=1 seconds=10
        t_update = @belapsed PointNeighbors.update!($nhs, $coords, $coords;
                                                    parallelization_backend = $backend) samples=20 evals=1 seconds=10
        dv = similar(coords)
        nbody_sweep!(dv, coords, nhs, backend)
        t_sweep = @belapsed nbody_sweep!($dv, $coords, $nhs, $backend) samples=20 evals=1 seconds=10
    end

    bytes = cell_storage_bytes(nhs_host)

    init_str = isnan(t_init) ? "    n/a   " : @sprintf("%9.3f ms", 1e3 * t_init)
    @printf("  %-30s mppc=%-5d  cell-bytes=%10d  init=%s  update=%9.3f ms  sweep=%9.3f ms\n",
            label, mppc, bytes, init_str, 1e3 * t_update, 1e3 * t_sweep)
    return (; mppc, bytes, t_init, t_update, t_sweep)
end

function run_all(n_points, search_radius)
    println("="^100)
    @printf("n_points = %d,  search_radius = %.4f\n", n_points, search_radius)
    println("="^100)

    coords_host = make_point_cloud(n_points)

    mppc_default = 50
    # NOTE: must use percentile=1.0, otherwise ~1% of cells can exceed the estimate
    # and FullGridCellList's DynamicVectorOfVectors backend will crash on pushat!.
    mppc_est = estimate_max_points_per_cell(coords_host, search_radius; percentile = 1.0)
    mppc_est_p99 = estimate_max_points_per_cell(coords_host, search_radius)  # default p99
    @printf("estimate_max_points_per_cell  p=1.00 -> %d  (safe upper bound)\n", mppc_est)
    @printf("estimate_max_points_per_cell  p=0.99 -> %d  (UNSAFE — may underestimate true max)\n",
            mppc_est_p99)
    @printf("default                              -> %d\n", mppc_default)

    # ---- CPU ----
    println("\n-- CPU (Array) --")
    cpu_default = bench_one("default", coords_host, search_radius, mppc_default,
                            PointNeighbors.default_backend(coords_host); on_gpu = false)
    cpu_est     = bench_one("estimated", coords_host, search_radius, mppc_est,
                            PointNeighbors.default_backend(coords_host); on_gpu = false)

    println()
    @printf("  memory savings: %.1f%%  (%.2f MiB -> %.2f MiB)\n",
            100 * (1 - cpu_est.bytes / cpu_default.bytes),
            cpu_default.bytes / 2^20, cpu_est.bytes / 2^20)
    @printf("  update! speedup:     %.2fx\n", cpu_default.t_update / cpu_est.t_update)
    @printf("  initialize! speedup: %.2fx\n", cpu_default.t_init   / cpu_est.t_init)
    @printf("  sweep speedup:       %.2fx\n", cpu_default.t_sweep  / cpu_est.t_sweep)

    # ---- GPU ----
    if HAS_CUDA
        println("\n-- GPU (CUDA) --")
        backend = CUDA.CUDABackend()
        gpu_default = bench_one("default", coords_host, search_radius, mppc_default, backend; on_gpu = true)
        gpu_est     = bench_one("estimated", coords_host, search_radius, mppc_est, backend; on_gpu = true)

        println()
        @printf("  memory savings: %.1f%%  (%.2f MiB -> %.2f MiB)\n",
                100 * (1 - gpu_est.bytes / gpu_default.bytes),
                gpu_default.bytes / 2^20, gpu_est.bytes / 2^20)
        @printf("  update! speedup:     %.2fx\n", gpu_default.t_update / gpu_est.t_update)
        @printf("  initialize! speedup: %.2fx\n", gpu_default.t_init   / gpu_est.t_init)
        @printf("  sweep speedup:       %.2fx\n", gpu_default.t_sweep  / gpu_est.t_sweep)
    else
        println("\n-- GPU: CUDA not functional, skipping --")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Moderate size that finishes in a few seconds per config
    run_all(50_000, 0.05)
    run_all(200_000, 0.03)
end
