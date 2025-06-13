module ConstructModule

using JSON
using ..FeaturesModule
using ..StrategyModule
using ..ActionsModule
using ..OptimizerModule
using Plots

export TradingConstruct, ExperimentResults, run_experiment, from_json, describe_experiment_results

@kwdef struct TradingConstruct
    
    
    feature_params::FeatureParams
    strategy_params::StrategyParams
    action_params::Action
    optimizer_params::GeneticParams
end

function get_prices()::Vector{Float64}
    bars_json::Vector = []
    open(joinpath(@__DIR__, "data", "btc_prices.json"), "r") do file
        bars_json = JSON.parse(read(file, String))
    end
    
    prices::Vector{Float64} = []
    for bar = reverse(bars_json)
        push!(prices, bar["price"])
    end
    return prices
end

@kwdef struct ExperimentResults
    is_backtest_results::BacktestResults
    oos_backtest_results::BacktestResults
    optimizer_results::GeneticResults
end

function run_experiment(construct::TradingConstruct)::ExperimentResults
    
    prices = get_prices()
    is_prices = prices[1:1000]
    oos_prices = prices[1000:end]

    inputs = [input_feature.feature for input_feature = construct.feature_params.features]

    function criterion(net::Network)::Float64
        signals = generate_signals(net, is_prices, construct.feature_params.features)

        return backtest(signals, is_prices)
    end

    opt = GeneticOptimizer(
        inputs=inputs,
        outputs=[DiscreteFeature(0.0, 1.0), DiscreteFeature(0.0, 1.0)],
        time_limit=120.0,
        eval_func=criterion,
        params=construct.optimizer_params,
        network_params=construct.strategy_params.network_params
    )
    init_optimizer!(opt)
    optimizer_results = optimize!(opt)

    best_net = best_network(opt)

    is_signals = generate_signals(best_net, is_prices, construct.feature_params.features)
    is_results = backtest(is_signals, is_prices, detailed_results=true)

    oos_signals = generate_signals(best_net, oos_prices, construct.feature_params.features)
    oos_results = backtest(oos_signals, oos_prices, detailed_results=true)

    return ExperimentResults(
        is_backtest_results=is_results,
        oos_backtest_results=oos_results,
        optimizer_results=optimizer_results
    )
end

function construct_from_json(construct_json::Dict{String, Any})::TradingConstruct
    optimizer_json = construct_json["optimizer"]
    optimizer_params = GeneticParams(
        population_size=optimizer_json["population size"],
        n_elites=optimizer_json["n elites"],
        sequence_length=optimizer_json["sequence length"],
        mutation_rate=optimizer_json["mutation rate"],
        crossover_rate=optimizer_json["crossover rate"],
        tournament_size=optimizer_json["tournament size"],
        mutation_delta=optimizer_json["mutation delta"],
        crossover_delta=optimizer_json["crossover delta"],
        tournament_delta=optimizer_json["tournament delta"],
        mutation_range=(optimizer_json["mutation range"][1], optimizer_json["mutation range"][2]),
        crossover_range=(optimizer_json["crossover range"][1], optimizer_json["crossover range"][2]),
        tournament_range=(optimizer_json["tournament range"][1], optimizer_json["tournament range"][2]),
        n_gram=optimizer_json["n length"],
        diversity_target=optimizer_json["diversity target"]
    )

    features::Vector{InputFeature} = []
    get_continuous_feature(feature_json::Dict{String, Any}) = ContinuousFeature(
        feature_json["min value"], feature_json["max value"],
        feature_json["resolution"]
    )
    for feature_json = construct_json["features"]
        if feature_json["feature"] == "log prices"
            push!(features, LogPrices(
                ohlc="close",
                feature=get_continuous_feature(feature_json)
            ))
        elseif feature_json["feature"] == "rsi"
            push!(features, RSI(
                window=feature_json["window"],
                feature=get_continuous_feature(feature_json)
            ))
        elseif feature_json["feature"] == "normalized sma"
            push!(features, NormalizedSMA(
                window=feature_json["window"],
                feature=get_continuous_feature(feature_json)
            ))
        end
    end


    strategy_json = construct_json["strategy"]
    network_params = NetworkParams(
        comparison_penalty=strategy_json["comparison penalty"],
        node_penalty=strategy_json["node penalty"],
        switch_penalty=strategy_json["node penalty"],
        default_output=strategy_json["default output"]
    )

    return TradingConstruct(
        optimizer_params=optimizer_params,
        feature_params=FeatureParams(
            features=features
        ),
        strategy_params=StrategyParams(
            network_params=network_params
        )
    )
end

function describe_experiment_results(results::ExperimentResults)::String
    io = IOBuffer()

    println(io, "In sample backtest results")
    println(io)

    println(io, "sharpe ratio: ", results.is_backtest_results.sharpe_ratio)
    println(io, "trades entered: ", results.is_backtest_results.entries)
    println(io, "average holding time: ", results.is_backtest_results.avg_holding_time)
    println(io, "is invalid?: ", results.is_backtest_results.is_invalid)
    println(io)

    println(io, "Out of sample backtest results")
    println(io)

    println(io, "sharpe ratio: ", results.oos_backtest_results.sharpe_ratio)
    println(io, "trades entered: ", results.oos_backtest_results.entries)
    println(io, "average holding time: ", results.oos_backtest_results.avg_holding_time)
    println(io, "is invalid?: ", results.oos_backtest_results.is_invalid)
    println(io)

    println(io, "Optimizer results")
    println(io)

    println(io, "best score: ", results.optimizer_results.best_score)
    println(io, "generations: ", results.optimizer_results.generations)
    println(io, "inflection points:")
    for inflection_point = results.optimizer_results.inflection_points
        println(io, "generation: ", inflection_point[1], ", new score: ", inflection_point[2])
    end


    println(io)
    println(io, "best network description:")
    println(io)
    println(io, describe_network(results.optimizer_results.best_network))

    return String(take!(io))
end

end