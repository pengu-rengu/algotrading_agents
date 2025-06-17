module FeaturesModule

using Statistics
export AbstractFeature, ContinuousFeature, DiscreteFeature, InputFeature, NormalizedBollingerBands, NormalizedSMA, NormalizedEMA, RSI, LogPrices, FeatureParams, get_values, units_to_value

abstract type AbstractFeature end

struct ContinuousFeature <: AbstractFeature
    min_value::Float64
    max_value::Float64
    resolution::Int
end

units_to_value(cf::ContinuousFeature, units::Int)::Float64 = 
cf.min_value + ((cf.max_value - cf.min_value) / cf.resolution) * units

struct DiscreteFeature <: AbstractFeature
    min_value::Float64
    max_value::Float64
    resolution::Int
    DiscreteFeature(min_value::Float64, max_value::Float64) = new(min_value, max_value,
    max_value - min_value)
end

units_to_value(df::DiscreteFeature, units::Int)::Float64 = df.min_value + units

function sma_helper(values::Vector{Float64}, window::Int)::Vector{Float64}
    sma_values::Vector{Float64} = [0.0 for _ = 1:window-1]

    for i = eachindex(values)[window:end]
        push!(sma_values, mean(@view values[i-window+1:i]))
    end

    return sma_values
end

function ema_helper(values::Vector{Float64}, window::Int, smoothing::Int)::Vector{Float64}
    ema_values::Vector{Float64} = [0.0 for _ = 1:window-1]
    
    ema_value = mean(@view values[1:window])

    for i = eachindex(values)[window:end]
        multiplier = (smoothing / (1 + window))
        ema_value = (values[i] * multiplier) + (ema_value * (1 - multiplier))
        push!(ema_values, ema_value)
    end

    return ema_values
end

abstract type InputFeature end

@kwdef struct LogPrices <: InputFeature
    ohlc::String
    feature::ContinuousFeature
end

function get_values(price_feature::LogPrices, prices::Vector{Float64})::Vector{Float64}
    return [0.0; log.(prices[2:end] ./ prices[1:end-1])]
end

@kwdef struct NormalizedSMA <: InputFeature
    window::Int
    feature::ContinuousFeature
end

function get_values(sma::NormalizedSMA, prices::Vector{Float64})::Vector{Float64}
    return sma_helper(prices, sma.window) ./ prices
end

@kwdef struct NormalizedEMA <: InputFeature
    window::Int
    smoothing::Int
    feature::ContinuousFeature
end

function get_values(ema::NormalizedEMA, prices::Vector{Float64})::Vector{Float64}
    return ema_helper(prices, ema.window, ema.smoothing) ./ prices
end

@kwdef struct RSI <: InputFeature
    window::Int
    smoothing::Int
    feature::ContinuousFeature
end

function get_values(rsi::RSI, prices::Vector{Float64})::Vector{Float64}
    rsi_values::Vector{Float64} = [0.0 for _ = 1:rsi.window-1]
    differences = [0; diff(prices)]
    ema_gains::Vector{Float64} = ema_helper(max.(differences, 0), rsi.window, rsi.smoothing)
    ema_losses::Vector{Float64} = ema_helper(max.(-differences, 0), rsi.window, rsi.smoothing)

    for i = eachindex(prices)[rsi.window:end]
        rs = ema_gains[i] / ema_losses[i]

        push!(rsi_values, 100 - (100 / (1 + rs)))
    end

    return rsi_values
end

@kwdef struct NormalizedBollingerBands <: InputFeature
    band::String
    window::Int
    std_multiplier::Float64
    feature::ContinuousFeature
end

function get_values(bb::NormalizedBollingerBands, prices::Vector{Float64})::Vector{Float64}
    sma_values = sma_helper(prices, bb.window)

    bb_values::Vector{Float64} = [0.0 for _ = 1:bb.window-1]

    for i = eachindex(prices)[bb.window:end]
        if bb.band == "upper"
            push!(bb_values, sma_values[i] + bb.std_multiplier * std(@view prices[i-bb.window+1:i]))
        elseif bb.band == "lower"
            push!(bb_values, sma_values[i] - bb.std_multiplier * std(@view prices[i-bb.window+1:i]))
        end
    end

    return bb_values ./ prices
end

@kwdef struct FeatureParams
    features::Vector{InputFeature}
end

end