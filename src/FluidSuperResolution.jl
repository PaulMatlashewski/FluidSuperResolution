module FluidSuperResolution

include("fluid.jl")
export Field, Velocity, Fluid, project!, advect!, divergence

include("data.jl")
export PerlinGrid, perlin_noise, sample

end # module
