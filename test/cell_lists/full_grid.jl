@testset "`FullGridCellList`" verbose=true begin
    # Test that an error is thrown when too many dimensions are used
    @testset "constructor" begin
        min_corner = zeros(101)
        max_corner = ones(101)
        search_radius = 1.0

        error_string = "FullGridCellList only supports up to 100 dimensions"
        @test_throws error_string FullGridCellList(; min_corner, max_corner)
        @test_throws error_string FullGridCellList(; min_corner, max_corner, search_radius)

        min_corner = zeros(3)
        max_corner = ones(2)
        error_string = "min_corner and max_corner must have the same length"
        @test_throws error_string FullGridCellList(; min_corner, max_corner)
    end

    # Test that `update!` throws an error when a particle is outside the bounding box
    @testset "`update!` bounds check" begin
        @testset "$(N)D" for N in 1:3
            min_corner = fill(0.0, N)
            max_corner = fill(10.0, N)
            search_radius = 1.0

            cell_list = FullGridCellList(; search_radius, min_corner, max_corner)

            # Introduce the same rounding errors for this to pass
            @test cell_list.min_corner == fill(-1.001, N)
            @test cell_list.max_corner == fill(10.0 + 1.001, N)

            nhs = GridNeighborhoodSearch{N}(; cell_list, search_radius)
            y = rand(N, 10)
            error_string = "particle coordinates are NaN or outside the domain bounds of the cell list"

            y[1, 7] = NaN
            @test_throws error_string initialize!(nhs, y, y)

            y[1, 7] = min_corner[1] - 0.01
            @test_throws error_string initialize!(nhs, y, y)

            # A bit more than max corner might still be inside the grid,
            # but one search radius more is always outside.
            # Also accounting for 0.001 extra padding (see above).
            y[1, 7] = max_corner[1] + 1.01
            @test_throws error_string initialize!(nhs, y, y)

            y[1, 7] = 0.0
            @trixi_test_nowarn initialize!(nhs, y, y)
            @trixi_test_nowarn update!(nhs, y, y)

            y[1, 7] = 10.0
            @trixi_test_nowarn update!(nhs, y, y)

            y[1, 7] = NaN
            @test_throws error_string update!(nhs, y, y)

            # A bit more than max corner might still be inside the grid,
            # but one search radius more is always outside.
            # Also accounting for 0.001 extra padding (see above).
            y[1, 7] = max_corner[1] + 1.01
            @test_throws error_string update!(nhs, y, y)

            y[1, 7] = min_corner[1] - 0.01
            @test_throws error_string update!(nhs, y, y)
        end
    end

    # Test adaptive allocation estimation
    @testset "`estimate_max_points_per_cell`" begin
        # Test 1: Uniform grid of points
        @testset "uniform grid" begin
            coords = zeros(3, 27)
            idx = 1
            for i in 1:3, j in 1:3, k in 1:3
                coords[:, idx] = [i, j, k]
                idx += 1
            end

            search_radius = 1.1
            estimated = estimate_max_points_per_cell(coords, search_radius)

            @test estimated >= 1
            @test estimated <= 10
        end

        # Test 2: Clustered points (sparse grid)
        @testset "clustered distribution" begin
            Random.seed!(42)
            n_points = 1000

            coords = zeros(3, n_points)

            # Dense cluster: 800 points in [0.4, 0.6]³
            for i in 1:800
                coords[:, i] = rand(3) .* 0.2 .+ 0.4
            end

            # Sparse: 200 points in [0, 1]³
            for i in 801:n_points
                coords[:, i] = rand(3)
            end

            search_radius = 0.05
            estimated = estimate_max_points_per_cell(coords, search_radius)

            @test estimated >= 1
            @test estimated < 100
        end

        # Test 3: Few points (3 points in 3D)
        @testset "few points" begin
            coords = [0.5 0.6 0.7;
                      0.5 0.6 0.7;
                      0.5 0.6 0.7]
            search_radius = 0.1

            estimated = estimate_max_points_per_cell(coords, search_radius)

            @test estimated >= 1
            @test estimated <= 5
        end

        # Test 4: Empty distribution
        @testset "empty distribution" begin
            coords = zeros(3, 0)
            search_radius = 0.1

            estimated = estimate_max_points_per_cell(coords, search_radius)

            @test estimated == 1
        end

        # Test 5: Zero search radius
        @testset "zero search radius" begin
            coords = randn(3, 100)
            search_radius = 0.0

            estimated = estimate_max_points_per_cell(coords, search_radius)

            @test estimated == 1
        end

        # Test 6: Single point (1 point in 3D)
        @testset "single point" begin
            coords = reshape([1.0, 2.0, 3.0], 3, 1)
            search_radius = 0.1

            estimated = estimate_max_points_per_cell(coords, search_radius)

            @test estimated >= 1
        end

        # Test 7: Percentile sensitivity
        @testset "percentile sensitivity" begin
            Random.seed!(123)
            coords = randn(2, 1000) .* 0.5
            search_radius = 0.2

            est_90 = estimate_max_points_per_cell(coords, search_radius; percentile=0.90)
            est_99 = estimate_max_points_per_cell(coords, search_radius; percentile=0.99)
            est_100 = estimate_max_points_per_cell(coords, search_radius; percentile=1.0)

            @test est_99 >= est_90
            @test est_100 >= est_99

            # Out-of-range percentile must be rejected
            @test_throws ArgumentError estimate_max_points_per_cell(coords, search_radius;
                                                                    percentile=0.0)
            @test_throws ArgumentError estimate_max_points_per_cell(coords, search_radius;
                                                                    percentile=1.5)
        end

        # Test 8: Safety margin
        @testset "safety margin impact" begin
            Random.seed!(456)
            coords = randn(2, 500)
            search_radius = 0.1

            est_1_0 = estimate_max_points_per_cell(coords, search_radius; safety_margin=1.0)
            est_1_5 = estimate_max_points_per_cell(coords, search_radius; safety_margin=1.5)

            @test est_1_5 >= est_1_0
        end

        # Test 9: Integration with FullGridCellList
        @testset "integration with FullGridCellList" begin
            Random.seed!(789)
            points = randn(3, 100)
            search_radius = 0.2

            estimated = estimate_max_points_per_cell(points, search_radius)

            cell_list = FullGridCellList(
                min_corner=(-2.0, -2.0, -2.0),
                max_corner=(2.0, 2.0, 2.0),
                search_radius=search_radius,
                max_points_per_cell=estimated
            )

            @test cell_list isa FullGridCellList
        end

        # Test 9b: Default estimate must be a hard upper bound — `initialize!`
        # used to throw a `BoundsError` from `pushat!` because the previous default
        # `percentile = 0.99` could underestimate the true maximum cell occupancy.
        @testset "default estimate is a safe upper bound" begin
            Random.seed!(2024)
            coords = rand(Float32, 3, 50_000)
            search_radius = 0.05f0

            estimated = estimate_max_points_per_cell(coords, search_radius)

            cell_list = FullGridCellList(; min_corner=(0.0f0, 0.0f0, 0.0f0),
                                         max_corner=(1.0f0, 1.0f0, 1.0f0),
                                         search_radius,
                                         max_points_per_cell=estimated)
            nhs = GridNeighborhoodSearch{3}(; search_radius,
                                            n_points=size(coords, 2),
                                            cell_list)
            # Must not throw — this is the regression check for the percentile bug.
            PointNeighbors.initialize!(nhs, coords, coords)
            @test true
        end

        # Test 10: Explicit bounds
        @testset "explicit domain bounds" begin
            coords = [0.25 0.75 0.5; 0.5 0.5 0.75]  # 2D points: (0.25,0.5), (0.75,0.5), (0.5,0.75)
            search_radius = 0.1

            estimated1 = estimate_max_points_per_cell(
                coords, search_radius;
                min_corner=[0.0, 0.0], max_corner=[1.0, 1.0]
            )

            estimated2 = estimate_max_points_per_cell(coords, search_radius)

            @test estimated1 >= 1
            @test estimated2 >= 1
        end

        # Test 11: Memory efficiency
        @testset "memory efficiency" begin
            Random.seed!(999)
            n_grid = 20
            domain_size = n_grid
            coords = rand(3, 10000) .* domain_size

            search_radius = 1.0

            estimated = estimate_max_points_per_cell(coords, search_radius)

            default_memory = 8000 * 100 * 4
            estimated_memory = 8000 * estimated * 4
            savings_percent = (1 - estimated_memory / default_memory) * 100

            @test savings_percent > 20
            @test estimated < 100
        end
    end
end
