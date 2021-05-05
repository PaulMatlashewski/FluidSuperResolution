module FluidSuperResolution

include("fluid.jl")
export Field, Velocity, Fluid, project!, advect!,  Dx, Dy, divergence, vorticity

include("data.jl")
export PerlinGrid, perlin_noise, perlin_velocity, sample, wrap

end # module
