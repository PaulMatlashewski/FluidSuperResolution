module FluidSuperResolution

include("fluid.jl")
export Field, Velocity, Fluid, project!, advect!,  Dx, Dy, divergence, vorticity, euler

include("perlin_velocity.jl")
export PerlinGrid, perlin_noise, perlin_velocity

include("poisson_disk_sample.jl")
export poisson_disk_sample

end # module
