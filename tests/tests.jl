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
