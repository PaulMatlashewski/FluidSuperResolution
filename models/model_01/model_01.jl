using FluidSuperResolution
using Flux
using Random
using writedlm
using BSON: @save

function model()
    Random.seed!(42)
    dt = 0.01    # Time step
    iters = 1000 # Training iterations
    η = 0.001    # Learning rate
    α = 1.0      # Divergence loss weight

    fluids = [
        Fluid(0.0, 64, 64),
        Fluid(0.0, 128, 128),
        Fluid(0.0, 256, 256)
    ]

    model = Flux.Chain(
        Conv((5, 5), 2=>16, stride=1, pad=SamePad(), relu),
        Conv((5, 5), 16=>32, stride=1, pad=SamePad(), relu),
        ConvTranspose((5, 6), 32=>32, stride=2, pad=(2, 2), relu),
        Conv((5, 5), 32=>16, stride=1, pad=SamePad(), relu),
        Conv((5, 5), 16=>2, stride=1, pad=SamePad())
    )
    θ = Flux.params(model)
    opt = ADAM(η)
    
    y1_interp = zeros(65, 64, 2, 6)
    y2_interp = zeros(129, 128, 2, 6)
    model_losses = []
    div_losses = []
    interp_losses = []
    local model_loss
    local div_loss
    for i in 1:iters
        println("Training step: $(i)")
        println("    Sampling")
        x, y1, y2 = sample_batch(fluids, dt)
        println("    Prediction")
        gs = gradient(θ) do
            ŷ1 = model(x)
            ŷ2 = model(ŷ1)
            model_loss = Flux.mse(ŷ1, y1) + Flux.mse(ŷ2, y2)
            div_loss = α * (divergence_loss(ŷ1) + divergence_loss(ŷ2))
            return model_loss + div_loss
        end
        Flux.Optimise.update!(opt, θ, gs)
        interpolation_loss!(y1_interp, y2_interp, x)
        interp_loss = Flux.mse(y1_interp, y1) + Flux.mse(y2_interp, y2)
        println("              ||∇||: $(norm(gs))")
        println("           Div Loss: $(div_loss)")
        println("         Model Loss: $(model_loss)")
        println("         Total Loss: $(model_loss + div_loss)")
        println(" Interpolation Loss: $(interp_loss)")
        push!(model_losses, model_loss)
        push!(div_losses, div_loss)
        push!(interp_losses, interp_loss)
    end
    @save "model.bson" model
    open("loss.txt", "w") do io
        writedlm(io, [model_losses; div_losses; interp_losses])
    end
end

function divergence_loss(y)
    n, m, _, batchsize = size(y)
    s = 0.0
    for i in 1:batchsize
        s += sum(divergence(y[:, :, 1, i], y[:, :, 2, i]').^2)
    end
    return s / (n * m)
end

function interpolation_loss!(y1_interp, y2_interp, x)
    batchsize = size(x, 4)
    for i in 1:batchsize
        x_interp_u = Field(x[:, :, 1, i], (1, 33), (1, 32), (0.0, 0.5))
        x_interp_v = Field(x[:, :, 2, i], (1, 33), (1, 32), (0.0, 0.5))
        y1_interp_u = Field(zeros(65, 64), (1, 65), (1, 64), (0.0, 0.5))
        y1_interp_v = Field(zeros(65, 64), (1, 65), (1, 64), (0.0, 0.5))
        y2_interp_u = Field(zeros(129, 128), (1, 129), (1, 128), (0.0, 0.5))
        y2_interp_v = Field(zeros(129, 128), (1, 129), (1, 128), (0.0, 0.5))
        interpolate!(y1_interp_u, x_interp_u)
        interpolate!(y2_interp_u, y1_interp_u)
        interpolate!(y1_interp_v, x_interp_v)
        interpolate!(y2_interp_v, y1_interp_v)
        y1_interp[:, :, 1, i] = y1_interp_u.values
        y1_interp[:, :, 2, i] = y1_interp_v.values
        y2_interp[:, :, 1, i] = y2_interp_u.values
        y2_interp[:, :, 2, i] = y2_interp_v.values
    end
end

function sample_batch(fluids, dt)
    # Samples from Perlin noise
    x_1, y1_1, y2_1 = sample_velocity_batch(fluids, 4, 0.5, 2, dt)
    x_2, y1_2, y2_2 = sample_velocity_batch(fluids, 4, 0.2, 5, dt)
    x_3, y1_3, y2_3 = sample_velocity_batch(fluids, 8, 0.2, 1, dt)
    x_4, y1_4, y2_4 = sample_velocity_batch(fluids, 8, 0.2, 5, dt)
    # Uniform flow sample
    u = rand() * 1.2
    v = rand() * 1.2
    x_5, y1_5, y2_5 = (cat(u * ones(33, 32, 2), v * ones(33, 32, 2), dims=3),
                       cat(u * ones(65, 64, 2), v * ones(65, 64, 2), dims=3),
                       cat(u * ones(129, 128, 2), v * ones(129, 128, 2), dims=3))
    x_6, y1_6, y2_6 = (zeros(33, 32, 4), zeros(65, 64, 4), zeros(129, 128, 4))
    # Compute increments
    dx_1 = cat(x_1[:, :, 3] - x_1[:, :, 1], x_1[:, :, 4] - x_1[:, :, 2], dims=3)
    dx_2 = cat(x_2[:, :, 3] - x_2[:, :, 1], x_2[:, :, 4] - x_2[:, :, 2], dims=3)
    dx_3 = cat(x_3[:, :, 3] - x_3[:, :, 1], x_3[:, :, 4] - x_3[:, :, 2], dims=3)
    dx_4 = cat(x_4[:, :, 3] - x_4[:, :, 1], x_4[:, :, 4] - x_4[:, :, 2], dims=3)
    dx_5 = cat(x_5[:, :, 3] - x_5[:, :, 1], x_5[:, :, 4] - x_5[:, :, 2], dims=3)
    dx_6 = cat(x_6[:, :, 3] - x_6[:, :, 1], x_6[:, :, 4] - x_6[:, :, 2], dims=3)
    dx = cat(dx_1, dx_2, dx_3, dx_4, dx_5, dx_6, dims=4)

    dy1_1 = cat(y1_1[:, :, 3] - y1_1[:, :, 1], y1_1[:, :, 4] - y1_1[:, :, 2], dims=3)
    dy1_2 = cat(y1_2[:, :, 3] - y1_2[:, :, 1], y1_2[:, :, 4] - y1_2[:, :, 2], dims=3)
    dy1_3 = cat(y1_3[:, :, 3] - y1_3[:, :, 1], y1_3[:, :, 4] - y1_3[:, :, 2], dims=3)
    dy1_4 = cat(y1_4[:, :, 3] - y1_4[:, :, 1], y1_4[:, :, 4] - y1_4[:, :, 2], dims=3)
    dy1_5 = cat(y1_5[:, :, 3] - y1_5[:, :, 1], y1_5[:, :, 4] - y1_5[:, :, 2], dims=3)
    dy1_6 = cat(y1_6[:, :, 3] - y1_6[:, :, 1], y1_6[:, :, 4] - y1_6[:, :, 2], dims=3)
    dy1 = cat(dy1_1, dy1_2, dy1_3, dy1_4, dy1_5, dy1_6, dims=4)

    dy2_1 = cat(y2_1[:, :, 3] - y2_1[:, :, 1], y2_1[:, :, 4] - y2_1[:, :, 2], dims=3)
    dy2_2 = cat(y2_2[:, :, 3] - y2_2[:, :, 1], y2_2[:, :, 4] - y2_2[:, :, 2], dims=3)
    dy2_3 = cat(y2_3[:, :, 3] - y2_3[:, :, 1], y2_3[:, :, 4] - y2_3[:, :, 2], dims=3)
    dy2_4 = cat(y2_4[:, :, 3] - y2_4[:, :, 1], y2_4[:, :, 4] - y2_4[:, :, 2], dims=3)
    dy2_5 = cat(y2_5[:, :, 3] - y2_5[:, :, 1], y2_5[:, :, 4] - y2_5[:, :, 2], dims=3)
    dy2_6 = cat(y2_6[:, :, 3] - y2_6[:, :, 1], y2_6[:, :, 4] - y2_6[:, :, 2], dims=3)
    dy2 = cat(dy2_1, dy2_2, dy2_3, dy2_4, dy2_5, dy2_6, dims=4)

    return dx, dy1, dy2
end
