using FluidSuperResolution
using Plots
import Base: copy

function simulation_test(n, m, dt)
    fluid = Fluid(0.0, n, m)
    ink = Field(0.0, (1, n), (1, m), (0.5, 0.5), n, m)

    # Apply Initial conditions
    source_start = Int(round(0.48 * n))
    source_end = Int(round(0.52 * n))
    source_val = 2.0 * m

    ink.values[1, source_start:source_end] .= source_val
    fluid.velocity.u.values[1, source_start:source_end] .= source_val

    gr(show=true)
    while true
        # Pressure projection
        project!(fluid, dt)
        # Advection
        advect!(ink, fluid, dt)
        advect!(fluid.velocity, fluid, dt)
        # Boundary conditions
        fluid.velocity.u.values[1, 1:source_start-1] .= 0.0
        fluid.velocity.u.values[1, source_start:source_end] .= source_val
        fluid.velocity.u.values[1, source_end+1:end] .= 0.0
        fluid.velocity.u.values[end, :] .= 0.0
        fluid.velocity.v.values[:, 1] .= 0.0
        fluid.velocity.v.values[:, end] .= 0.0
        ink.values[1, source_start:source_end] .= source_val
        # Plot
        display(heatmap(ink.values',
                        clim=(0, n),
                        aspect_ratio=:equal,
                        xlims=(0, n+1), ylims=(0, n+1),
                        axis=false,
                        grid=false,
                        labels=false))
    end
end

# Divergence of velocity should be 0
function perlin_velocity_test(grid, n, p, m)
    u_div = zeros(n, n)
    v_div = zeros(n, n)
    u_offset = [0.0, 0.5]
    v_offset = [0.5, 0.0]
    ∇(x) = ForwardDiff.gradient(x -> sample(x, grid, p, m, n), x)
    vel(x) = [∇(x)[2], -∇(x)[1]]
    div_curl(x) = ForwardDiff.gradient(x -> vel(x)[1], x)[1] + ForwardDiff.gradient(x -> vel(x)[2], x)[2]
    for j in 1:n
        for i in 1:n
            x = [i, j]
            u_div[i, j] = div_curl(x + u_offset)
            v_div[i, j] = div_curl(x + v_offset)
        end
    end
    return u_div, v_div
end

## Streamplots
struct Particles{T}
    xs::Vector{T}
    ys::Vector{T}
    original_xs::Vector{T}
    original_ys::Vector{T}
    age_offsets::Vector{Int}
    ages::Vector{Int}
end

function Particles(spacing, frames::Int)
    xs, ys = poisson_disk_sample(spacing, 30)
    n = length(xs)
    age_offsets = rand(1:frames, n)
    ages = zeros(Int, n)
    return Particles(xs, ys, copy(xs), copy(ys), age_offsets, ages)
end

function Base.copy(particles::Particles{T}) where {T}
    return Particles(copy(particles.xs), copy(particles.ys),
                     copy(particles.original_xs), copy(particles.original_ys),
                     copy(particles.age_offsets), copy(particles.ages))
end

function advect!(particles, velocity, dt)
    # Convert particles coordinates to velocity grid coordinates
    n, m = size(velocity.u.values, 2), size(velocity.v.values, 1)
    x0, y0 = particles.xs * n .+ 1, particles.ys * m .+ 1
    x1, y1 = euler(x0, y0, velocity, dt)
    particles.xs .= (x1 .- 1) / n
    particles.ys .= (y1 .- 1) / m
end

function streamplot(velocity, spacing, duration, fps, tail_duration, dt, file)
    frames = duration * fps
    tail_frames = Int(ceil(tail_duration * fps))
    particles = Particles(spacing, frames)
    particle_set = [copy(particles) for _ in 1:tail_frames]
    # Set up blur plot
    for i in 1:tail_frames
        for j in 1:i
            advect!(particle_set[j], velocity, dt)
        end
    end
    xs = []
    ys = []
    as = []
    anim = @animate for t in 1:frames
        xs = []
        ys = []
        as = []
        for (i, particles) in enumerate(particle_set)
            advect!(particles, velocity, dt)
            a = (length(particle_set) - i) / (length(particle_set) - 1)
            xs = vcat(xs, particles.xs)
            ys = vcat(ys, particles.ys)
            as = vcat(as, [a for _ in 1:length(particles.xs)])
        end
        scatter(xs, ys, grid=false, axis=false, xlims=(0, 1), ylims=(0, 1), labels=false,
                markerstrokewidth=0, markercolor=1, markeralpha=as, markersize=1)
    end
    gif(anim, file, fps=fps)
end
