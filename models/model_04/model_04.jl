using FluidSuperResolution
using LinearAlgebra
using Flux
using Random
using BSON
using BSON: @save
using Statistics
using Plots

function model()
    Random.seed!(42)
    dt = 0.01        # Time step
    iters = 200      # Training epochs
    sample_steps = 5 # Steps per epoch
    η = 0.001        # Learning rate
    α = 1.0          # Divergence loss weight

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

    losses = Dict(
        "model_loss_1" => [],
        "model_loss_2" => [],
        "div_loss_1" => [],
        "div_loss_2" => [],
        "interp_loss_1" => [],
        "interp_loss_2" => [],
        "interp_div_loss_1" => [],
        "interp_div_loss_2" => [],
        "gradient_size" => []
    )
    local model_loss_1
    local model_loss_2
    local div_loss_1
    local div_loss_2
    for i in 1:iters
        println("Training step: $(i)")
        println("    Sampling Batch")
        x, y1, y2 = sample_batch(fluids, dt)
        # Take multiple gradient steps for each sample
        for j in 1:sample_steps
            println("    Prediction $(j) of $(sample_steps)")
            gs = gradient(θ) do
                ŷ1 = model(x)
                ŷ2 = model(ŷ1)
                model_loss_1 = Flux.mse(ŷ1, y1)
                model_loss_2 = Flux.mse(ŷ2, y2)
                div_loss_1 = α * divergence_loss(ŷ1)
                div_loss_2 = α * divergence_loss(ŷ2)
                return model_loss_1 + model_loss_2 + div_loss_1 + div_loss_2
            end
            Flux.Optimise.update!(opt, θ, gs)
            # Interpolation losses for comparison
            interpolation_loss!(y1_interp, y2_interp, x)
            interp_loss_1 = Flux.mse(y1_interp, y1)
            interp_loss_2 = Flux.mse(y2_interp, y2)
            interp_div_loss_1 = α * divergence_loss(y1_interp)
            interp_div_loss_2 = α * divergence_loss(y2_interp)
            println("              ||∇||: $(norm(gs))")
            println("           Div Loss: $(div_loss_1 + div_loss_2)")
            println("    Interp Div Loss: $(interp_div_loss_1 + interp_div_loss_2)")
            println("         Model Loss: $(model_loss_1 + model_loss_2)")
            println(" Interpolation Loss: $(interp_loss_1 + interp_loss_2)")
            push!(losses["model_loss_1"], model_loss_1)
            push!(losses["model_loss_2"], model_loss_2)
            push!(losses["div_loss_1"], div_loss_1)
            push!(losses["div_loss_2"], div_loss_2)
            push!(losses["interp_loss_1"], interp_loss_1)
            push!(losses["interp_loss_2"], interp_loss_2)
            push!(losses["interp_div_loss_1"], interp_div_loss_1)
            push!(losses["interp_div_loss_2"], interp_div_loss_2)
            push!(losses["gradient_size"], norm(gs))
        end
    end
    @save "model_4.bson" model
    @save "loss_4.bson" losses
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
    dx = cat([velocity_increment(x) for x in [x_1, x_2, x_3, x_4, x_5, x_6]]..., dims=4)
    dy1 = cat([velocity_increment(y) for y in [y1_1, y1_2, y1_3, y1_4, y1_5, y1_6]]..., dims=4)
    dy2 = cat([velocity_increment(y) for y in [y2_1, y2_2, y2_3, y2_4, y2_5, y2_6]]..., dims=4)

    return dx, dy1, dy2
end

function velocity_increment(x)
    return cat(x[:, :, 3] - x[:, :, 1], x[:, :, 4] - x[:, :, 2], dims=3)
end

function post_process()
    model = BSON.load("model_4.bson")[:model]
    losses = BSON.load("loss_4.bson")[:losses]

    model_loss = losses["model_loss_1"] + losses["model_loss_2"]
    interp_loss = losses["interp_loss_1"] + losses["interp_loss_2"]
    interp_loss_mean = mean(interp_loss)
    plot(model_loss, label="model loss", size=(500, 500))
    hline!([interp_loss_mean], label="Interpolation loss")
    Plots.pdf("model_loss.pdf")

    fluids = [
        Fluid(0.0, 64, 64),
        Fluid(0.0, 128, 128),
        Fluid(0.0, 256, 256)
    ]
    dt = 0.01
    x, y1, y2 = sample_batch(fluids, dt)
    ŷ1 = model(x)
    ŷ2 = model(ŷ1)
    y1_interp = zeros(65, 64, 2, 6)
    y2_interp = zeros(129, 128, 2, 6)
    interpolation_loss!(y1_interp, y2_interp, x)
    batchsize = size(x, 4)
    for i in 1:batchsize
        clims = (minimum(y2[:, :, 1, i]), maximum(y2[:, :, 1, i]))
        heatmap(x[:, :, 1, i], clims=clims, title="observation 32x32", size=(500,500), showaxis = false, grid=false, axis=nothing)
        png("x_$(i).png")
        heatmap(y1[:, :, 1, i], clims=clims, title="simulation 64x64", size=(500,500), showaxis = false, grid=false, axis=nothing)
        png("y1_$(i).png")
        heatmap(y2[:, :, 1, i], clims=clims, title="simulation 128x128", size=(500,500), showaxis = false, grid=false, axis=nothing)
        png("y2_$(i).png")
        heatmap(ŷ1[:, :, 1, i], clims=clims, title="model 64x64", size=(500,500), showaxis = false, grid=false, axis=nothing)
        png("y1_hat_$(i).png")
        heatmap(ŷ2[:, :, 1, i], clims=clims, title="model 128x128", size=(500,500), showaxis = false, grid=false, axis=nothing)
        png("y2_hat_$(i).png")
        heatmap(y1_interp[:, :, 1, i], clims=clims, title="interpolation 64x64", size=(500,500), showaxis = false, grid=false, axis=nothing)
        png("y1_interp_$(i).png")
        heatmap(y2_interp[:, :, 1, i], clims=clims, title="interpolation 128x128", size=(500,500), showaxis = false, grid=false, axis=nothing)
        png("y2_interp_$(i).png")
    end
end
