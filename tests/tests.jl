using FluidSuperResolution
using Plots

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
