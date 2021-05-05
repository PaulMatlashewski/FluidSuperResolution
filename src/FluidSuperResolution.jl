module FluidSuperResolution

include("fluid.jl")
export Field, Velocity, Fluid, project!, advect!, divergence, Dx, Dy

include("data.jl")
export PerlinGrid, perlin_noise, perlin_velocity, sample, wrap, wrap2

end # module
