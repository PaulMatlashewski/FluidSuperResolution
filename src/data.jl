using LinearAlgebra
using ForwardDiff
using DiffRules: @define_diffrule

struct PerlinGrid{T}
    n::Int
    d::Array{T, 3}
end

function PerlinGrid(n)
    θ = rand(n + 1, n + 1) * 2π
    d = zeros(n + 1, n + 1, 2)
    d[:, :, 1] = cos.(θ)
    d[:, :, 2] = sin.(θ)
    d[end, :, :] .= d[1, :, :]
    d[:, end, :] .= d[:, 1, :]
    return PerlinGrid(n, d)
end

function smoothstep(x)
    return 6x^5 - 15x^4 + 10x^3
end

function smoothstep(x, y)
    return smoothstep(x) * smoothstep(y)
end

# AD Friendly mod
function wrap(x, n)
    return mod(x - 1, n) + 1
end

function wrap(x::ForwardDiff.Dual{T,V,P}, n::T2) where {T,V,P,T2}
    val = wrap(ForwardDiff.value(x), n)
    p = ForwardDiff.partials(x)
    return ForwardDiff.Dual{T}(val, p)
end

function sample(x, grid)
    i1 = Int(trunc(x[1]))
    j1 = Int(trunc(x[2]))
    i2 = min(i1 + 1, grid.n + 1)
    j2 = min(j1 + 1, grid.n + 1)

    dx = x - [i1, j1]
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

function sample(x, grid, p, m, n)
    x0 = (x .- 1) / n * grid.n .+ 1 # Scale to Perlin grid
    value = 0.0
    for k in 0:(m-1)
        x̂ = wrap.(2^k * x0, grid.n + 1)
        value += p^k * sample(x̂, grid)
    end
    return value
end

function perlin_noise(grid, n, p, m)
    values = zeros(n, n)
    for j in 1:n
        for i in 1:n
            x = [i + 0.5, j + 0.5]
            values[i, j] = sample(x, grid, p, m, n)
        end
    end
    return values
end

function perlin_velocity(grid, n, p, m)
    u = zeros(n + 1, n)
    v = zeros(n, n + 1)
    u_offset = [0.0, 0.5]
    v_offset = [0.5, 0.0]
    ∇(x) = ForwardDiff.gradient(x -> sample(x, grid, p, m, n), x)
    for j in 1:n
        for i in 1:n
            x = [i, j]
            u[i, j] = ∇(x + u_offset)[2] * n
            v[i, j] = -∇(x + v_offset)[1] * n
        end
    end
    for k in 1:n
        u[n + 1, k] = ∇([n + 1, k] + u_offset)[2] * n
        v[k, n + 1] = -∇([k, n + 1] + v_offset)[1] * n
    end
    return Velocity(u, v)
end
