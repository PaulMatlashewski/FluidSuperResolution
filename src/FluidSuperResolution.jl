module FluidSuperResolution

include("fluid.jl")
export Field, Velocity, Fluid, project!, advect!,  Dx, Dy, divergence, vorticity

include("data.jl")
export PerlinGrid, perlin_noise, perlin_velocity, sample, wrap

include("poisson_disk_sample.jl")
export poisson_disk_sample

end # module
