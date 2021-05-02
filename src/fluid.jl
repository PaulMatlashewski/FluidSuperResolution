using LinearAlgebra
using SparseArrays
using SuiteSparse.CHOLMOD
using SuiteSparse

struct Field{T}
    values::Matrix{T}
    offset::Tuple{T, T}
    xlims::Tuple{T, T}
    ylims::Tuple{T, T}
end

struct Velocity{T}
    u::Field{T}
    v::Field{T}
end

struct Fluid{T}
    velocity::Velocity{T}
    A::SuiteSparse.CHOLMOD.Factor{T}
    b::Vector{T}
    p::Vector{T}
    xlims::Tuple{T, T}
    ylims::Tuple{T, T}
    size::Tuple{Int, Int}
end

function Fluid(u, v, xlims, ylims, size)
    n, m = size
    dx = (xlims[2] - xlims[1]) / n
    dy = (ylims[2] - ylims[1]) / m

    u_ylims = (ylims[1], ylims[2] - dy)
    v_xlims = (xlims[1], xlims[2] - dx)
    u_field = Field(u, (0.0, 0.5), xlims, u_ylims)
    v_field = Field(v, (0.5, 0.0), v_xlims, ylims)
    velocity = Velocity(u_field, v_field)

    T = typeof(u[1, 1])
    A = SuiteSparse.CHOLMOD.cholesky(∇²(n, m ,T))
    b = zeros(n * m)
    p = zeros(n * m)

    return Fluid(velocity, A, b, p, xlims, ylims, size)
end

function Dx(values)
    return @views values[2:end, :] - values[1:end-1, :]
end

function Dy(values)
    return Dx(values')'
end

function divergence(velocity)
    Dx(velocity.u.values) .+ Dy(velocity.v.values)
end

function apply_pressure_gradient!(fluid, dt)
    n, m = fluid.size
    for j in 1:n
        for i in 1:n
            p = fluid.p[n * (j - 1) + i] * dt
            fluid.velocity.u.values[i, j] -= p
            fluid.velocity.v.values[i, j] -= p
            fluid.velocity.u.values[i + 1, j] += p
            fluid.velocity.v.values[i, j + 1] += p
        end
    end
end

function project!(fluid, dt)
    fluid.b .= (-divergence(fluid.velocity) / dt)[:]
    fluid.p .= fluid.A \ fluid.b
    apply_pressure_gradient!(fluid, dt)
end

function ∇²(n, m, T)
    k = 1
    I = []
    J = []
    V = T[]
    for i in 1:n
        for j in 1:m
            diag = 4 * one(T)
            if i > 1
                k_n = k - n
                push!(I, k_n)
                push!(J, k)
                push!(V, -one(T))
                # diag += one(T)
            end
            if i < n
                k_n = k + n
                push!(I, k_n)
                push!(J, k)
                push!(V, -one(T))
                # diag += one(T)
            end
            if j > 1
                k_n = k - 1
                push!(I, k_n)
                push!(J, k)
                push!(V, -one(T))
                # diag += one(T)
            end
            if j < n
                k_n = k + 1
                push!(I, k_n)
                push!(J, k)
                push!(V, -one(T))
                # diag += one(T)
            end
            push!(I, k)
            push!(J, k)
            push!(V, diag + 1e-10*one(T))
            k += 1
        end
    end
    return sparse(I, J, V)
end





function interpolate(x, y, u, offset)
    xmax, ymax = size(u)
    n, m = size(x)
    x = x[:]
    y = y[:]
    u = u[:]

    x_grid = min.(max.(x .- offset[1], 1), xmax)
    y_grid = min.(max.(y .- offset[2], 1), ymax)

    dx = mod.(x_grid, 1.0)
    dy = mod.(y_grid, 1.0)

    x1 = Int.(trunc.(x_grid))
    y1 = Int.(trunc.(y_grid))

    x2 = min.(x1 .+ 1, xmax)
    y2 = min.(y1 .+ 1, ymax)

    k1 = xmax * (y1 .- 1) .+ x1
    k2 = xmax * (y1 .- 1) .+ x2
    k3 = xmax * (y2 .- 1) .+ x1
    k4 = xmax * (y2 .- 1) .+ x2

    u1 = getindex(u, k1) .* (1 .- dx) .+ getindex(u, k2) .* dx
    u2 = getindex(u, k3) .* (1 .- dx) .+ getindex(u, k4) .* dx

    return reshape(u1 .* (1 .- dy) .+ u2 .* dy, (n, m))
end