module FeaturesModule

using ..NetworkModule
using Statistics

export InputFeature, NormalizedSMA, NormalizedEMA, RSI, LogPrices, 
FeatureParams, get_values

abstract type InputFeature end

function sma_helper()
    
end

function ema_helper()

end

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
    sma_values::Vector{Float64} = [0.0 for _ = 1:sma.window-1]

    for i = eachindex(prices)[sma.window:end]
        push!(sma_values, mean(prices[i-sma.window+1:i]))
    end

    return sma_values ./ prices
end

@kwdef struct NormalizedEMA <: InputFeature
    window::Int
    smoothing::Int
    feature::ContinuousFeature
end

function get_values(ema::NormalizedEMA, prices::Vector{Float64})::Vector{Float64}

    ema_values::Vector{Float64} = [0.0 for _ = 1:ema.window-1]
    
    ema_value = mean(prices[1:ema.window])

    for i = eachindex(prices)[ema.window:end]
        multiplier = (ema.smoothing / (1 + ema.window))
        ema_value = (prices[i] * multiplier) + (ema_value * (1 - multiplier))
        push!(ema_values, ema_value)
    end

    return ema_values ./ prices
end

@kwdef struct RSI <: InputFeature
    window::Int
    feature::ContinuousFeature
end

function get_values(rsi::RSI, prices::Vector{Float64})::Vector{Float64}
    rsi_values::Vector{Float64} = [0.0 for _ = 1:rsi.window-1]
    differences = [0; prices[2:end] - prices[1:end-1]]

    for i = eachindex(prices)[rsi.window:end]
        window_differences = differences[i-rsi.window+1:i]
        avg_gain = sum(max.(window_differences, 0)) / rsi.window
        avg_loss = sum(max.(-window_differences, 0)) / rsi.window

        rs = avg_gain / avg_loss

        push!(rsi_values, 100 - (100 / (1 + rs)))
    end

    return rsi_values
end

@kwdef struct FeatureParams
    features::Vector{InputFeature}
end

end