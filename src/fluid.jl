using LinearAlgebra
using SparseArrays
using SuiteSparse.CHOLMOD
using SuiteSparse

struct Field{T}
    values::Matrix{T}
    offset::Tuple{T, T}
    x::Matrix{T}
    y::Matrix{T}
end

function Field(values::Matrix{T}, xlims::Tuple{Int, Int}, ylims::Tuple{Int, Int}, offset::Tuple{T, T}) where {T}
    x = similar(values)
    y = similar(values)
    for (j, jval) in enumerate(ylims[1]:ylims[2])
        yval = jval + offset[2]
        for (i, ival) in enumerate(xlims[1]:xlims[2])
            xval = ival + offset[1]
            x[i, j] = xval
            y[i, j] = yval
        end
    end
    return Field(values, offset, x, y)
end

function Field(value::T, xlims::Tuple{Int, Int}, ylims::Tuple{Int, Int}, offset::Tuple{T, T}, n, m) where {T}
    return Field(value * ones(T, n, m), xlims, ylims, offset)
end

struct Velocity{T}
    u::Field{T}
    v::Field{T}
end

function Velocity(u_values::Matrix{T}, v_values::Matrix{T}) where {T}
    n = size(u_values)[2]
    u = Field(u_values, (1, n+1), (1, n), (zero(T), T(0.5)))
    v = Field(v_values, (1, n), (1, n+1), (T(0.5), zero(T)))
    return Velocity(u, v)
end

function Velocity(value::T, n::Int, m::Int) where {T}
    return Velocity(value * ones(T, n+1, m), value * ones(T, n, m+1))
end

struct Fluid{T}
    velocity::Velocity{T}
    A::SuiteSparse.CHOLMOD.Factor{T}
    b::Vector{T}
    p::Vector{T}
    size::Tuple{Int, Int}
end

function Fluid(value::T, n, m) where {T}
    velocity = Velocity(value, n, m)
    A = SuiteSparse.CHOLMOD.cholesky(∇²(n, m ,T))
    b = zeros(n * m)
    p = zeros(n * m)
    return Fluid(velocity, A, b, p, (n, m))
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
    
    u_flat = u[:]

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

    u1 = getindex(u_flat, k1) .* (1 .- dx) .+ getindex(u_flat, k2) .* dx
    u2 = getindex(u_flat, k3) .* (1 .- dx) .+ getindex(u_flat, k4) .* dx

    return u1 .* (1 .- dy) .+ u2 .* dy
end

function euler(x, y, u, v, dt)
    u_interp = interpolate(x, y, u, [0.0, 0.5])
    v_interp = interpolate(x, y, v, [0.5, 0.0])
    return (x + u_interp * dt, y + v_interp * dt)
end

function euler(x, y, velocity, dt)
    u_interp = interpolate(x, y, velocity.u.values, velocity.u.offset)
    v_interp = interpolate(x, y, velocity.v.values, velocity.v.offset)
    return (x + u_interp * dt, y + v_interp * dt)
end

# Semi-Lagrangian advection
function advect!(field::Field, fluid, dt)
    x, y = euler(field.x[:], field.y[:], fluid.velocity, -dt)
    field.values .= reshape(interpolate(x, y, field.values, field.offset), size(field.values))
end

function advect!(velocity::Velocity, fluid, dt)
    ux, uy = euler(velocity.u.x[:], velocity.u.y[:], fluid.velocity, -dt)
    vx, vy = euler(velocity.v.x[:], velocity.v.y[:], fluid.velocity, -dt)
    velocity.u.values .= reshape(interpolate(ux, uy, velocity.u.values, [0.0, 0.5]), size(velocity.u.values))
    velocity.v.values .= reshape(interpolate(vx, vy, velocity.v.values, [0.5, 0.0]), size(velocity.v.values))
end
