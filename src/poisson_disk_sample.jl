using LinearAlgebra

# Bridson, 2007. Fast Poisson Disk Sampling in Arbitrary Dimensions.
function poisson_disk_sample(r, max_check)
    # Initialize the particles with a random point in the unit square.
    xs = [rand()]
    ys = [rand()]
    # The cell size of the grid should be bounded by r / sqrt(2) so
    # that each grid cell will contain at most one sample
    n = Int(ceil(sqrt(2) / r))
    # Spatial grid for fast neighbor checking. The grid integers are
    # the particle indices
    grid = zeros(Int, n, n)
    mark_grid(grid, xs[1], ys[1], 1)
    # Active point list to generate new samples
    active_points = [1]
    while length(active_points) > 0
        k_id = rand(1:length(active_points))
        k = active_points[k_id]
        fails = 0
        while fails < max_check
            x, y = random_point_in_annulus(xs[k], ys[k], r, 2r)
            # Continue if point is outside domain
            if (x < 0) || (x > 1) || (y < 0) || (y > 1)
                fails += 1
                continue
            end
            # Continue if point is in an already visited grid cell
            if grid[index(grid, x, y)...] > 0
                fails += 1
                continue
            end
            # Check if point is within distance r of existing points
            valid = true
            for (i, j) in neighbors(grid, x, y)
                if grid[i, j] > 0
                    neighbor_point = [xs[grid[i, j]], ys[grid[i, j]]]
                    if norm(neighbor_point - [x, y]) < r
                        valid = false
                        fails += 1
                        break
                    end
                end
            end
            if valid
                push!(xs, x)
                push!(ys, y)
                grid[index(grid, x, y)...] = length(xs)
                push!(active_points, length(xs))
                # Reset failed attempts
                fails = 0
            end
        end
        deleteat!(active_points, k_id)
    end
    return xs, ys
end

function index(grid, x, y)
    n, m = size(grid)
    return Int(floor(x * n)) + 1, Int(floor(y * m)) + 1
end

function ingrid(grid, i, j)
    n, m = size(grid)
    return (i >= 1) && (i <= n) && (j >= 1) && (j <= m)
end

# Mark the grid cell containing x, y with the index k
function mark_grid(grid, x, y, k)
    i, j = index(grid, x, y)
    grid[i, j] = k
end

function random_point_in_annulus(x, y, r1, r2)
    r = rand() * (r2 - r1) + r1
    θ = rand() * 2π
    return x + r * cos(θ), y + r * sin(θ)
end

function neighbors(grid, x, y)
    i, j = index(grid, x, y)
    # Include 8 adjacent and diagonal neighbors
    idxs = [(i-1, j-1), (i+0, j-1), (i+1, j-1),
            (i-1, j+0),             (i+1, j+0),
            (i-1, j+1), (i+0, j+1), (i+1, j+1)]
    # Remove points outside grid
    filter!(idx -> ingrid(grid, idx...), idxs)
    return idxs
end
