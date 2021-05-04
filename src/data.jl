using LinearAlgebra

struct PerlinGrid{T}
    n::Int
    d::Array{T, 3}
end

function PerlinGrid(n)
    θ = rand(n, n) * 2π
    d = zeros(n, n, 2)
    d[:, :, 1] = cos.(θ)
    d[:, :, 2] = sin.(θ)
    return PerlinGrid(n, d)
end

function sample(grid, x)
    i1 = Int(trunc(x[1]))
    j1 = Int(trunc(x[2]))
    i2 = mod(i1, grid.n) + 1
    j2 = mod(j1, grid.n) + 1

    dx = mod.(x, 1.0)
    v1 = dot(grid.d[i1, j1, :], dx - [0, 0])
    v2 = dot(grid.d[i2, j1, :], dx - [1, 0])
    v3 = dot(grid.d[i2, j2, :], dx - [1, 1])
    v4 = dot(grid.d[i1, j2, :], dx - [0, 1])

    u1 = v1 * smoothstep(1 - dx[1], 1 - dx[2])
    u2 = v2 * smoothstep(dx[1], 1 - dx[2])
    u3 = v3 * smoothstep(dx[1], dx[2])
    u4 = v4 * smoothstep(1 - dx[1], dx[2])

    return u1 + u2 + u3 + u4
end

function smoothstep(x)
    return 6x^5 -15x^4 + 10x^3
end

function smoothstep(x, y)
    return smoothstep(x) * smoothstep(y)
end

function perlin_noise(grid, n, p, m)
    values = zeros(n, n)
    r = n / grid.n
    for j in 1:n
        for i in 1:n
            x = [i/r + 0.5, j/r + 0.5]
            for k in 0:(m-1)
                x̂ = mod.(2^k * x .- 1, grid.n) .+ 1
                values[i, j] += p^(k-1) * sample(grid, x̂)
            end
        end
    end
    return values
end
