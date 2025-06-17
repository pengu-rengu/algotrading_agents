module ConstructModule

using JSON
using ..FeaturesModule
using ..StrategyModule
using ..ActionsModule
using ..OptimizerModule

export TradingConstruct, ExperimentResults, run_experiment, describe_experiment_results, construct_from_json, get_prices

@kwdef struct TradingConstruct
    feature_params::FeatureParams
    strategy_params::StrategyParams
    actions_params::ActionsParams
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

    features = [input_feature.feature for input_feature = construct.feature_params.features]

    function criterion(net::Network)::Float64
        signals = generate_signals(net, is_prices, construct.feature_params.features)

        return backtest(signals, is_prices, construct.strategy_params) - get_penalty(net, construct.strategy_params.network_penalties)
    end

    opt = GeneticOptimizer(
        features=features,
        time_limit=120.0,
        eval_func=criterion,
        params=construct.optimizer_params,
        action_params=construct.actions_params,
    )
    init_optimizer!(opt)
    optimizer_results = optimize!(opt)

    best_net = best_network(opt)

    is_signals = generate_signals(best_net, is_prices, construct.feature_params.features)
    is_results = backtest(is_signals, is_prices, construct.strategy_params, detailed_results=true)

    oos_signals = generate_signals(best_net, oos_prices, construct.feature_params.features)
    oos_results = backtest(oos_signals, oos_prices, construct.strategy_params, detailed_results=true)

    return ExperimentResults(
        is_backtest_results=is_results,
        oos_backtest_results=oos_results,
        optimizer_results=optimizer_results
    )
end

function check_probs(probs::Vector{Float64}, expected_length::Int)
    return sum(probs) == 1.0 && length(probs) == expected_length
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
        operator_probs=optimizer_json["operator probs"] == "uniform" ? nothing : optimizer_json["operator probs"],
        n_gram=optimizer_json["n length"],
        diversity_target=optimizer_json["diversity target"]
    )

    if !isnothing(optimizer_params.operator_probs) && !check_probs(optimizer_params.operator_probs, actions_count)
        throw(ErrorException(string("Operator probabilities array length must be 3 and all probabilities must sum to 1")))
    end

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
                smoothing=feature_json["smoothing"],
                feature=get_continuous_feature(feature_json)
            ))
        elseif feature_json["feature"] == "normalized sma"
            push!(features, NormalizedSMA(
                window=feature_json["window"],
                feature=get_continuous_feature(feature_json)
            ))
        elseif feature_json["feature"] == "normalized ema"
            push!(features, NormalizedEMA(
                window=feature_json["window"],
                smoothing=feature_json["smoothing"],
                feature=get_continuous_feature(feature_json)
            ))
        elseif feature_json["feature"] == "normalized bollinger bands"
            push!(features, NormalizedBollingerBands(
                window=feature_json["window"],
                band=feature_json["band"],
                std_multiplier=feature_json["std multiplier"],
                feature=get_continuous_feature(feature_json)
            ))
        else
            throw(ErrorException("Unrecognized feature: " * feature_json["feature"]))
        end
    end

    actions_json = construct_json["actions"]
    meta_actions = Vector{Pair{String, Vector}}()
    for meta_action_json = actions_json["meta actions"]
        push!(meta_actions, meta_action_json["name"] => meta_action_json["sub actions"])
    end

    
    actions_params = ActionsParams(
        meta_actions=Dict(meta_actions),
        scaffolding=actions_json["scaffolding"],
        allow_functions=actions_json["allow functions"],
        allow_recurrence=actions_json["allow recurrence"],
        action_probs=actions_json["action probs"] == "uniform" ? nothing : actions_json["action probs"]
    )

    actions_count = length(all_actions(actions_params))
    if !isnothing(actions_params.action_probs) && !check_probs(actions_params.action_probs, actions_count)
        throw(ErrorException(string("Action probabilities array length must equal number of all available actions and meta actions (", actions_count, ") and all probabilities must sum to 1")))
    end

    strategy_json = construct_json["strategy"]
    penalties_json = strategy_json["penalties"]

    strategy_params=StrategyParams(
        network_penalties=NetworkPenalties(
            comparison_penalty=penalties_json["comparison penalty"],
            node_penalty=penalties_json["node penalty"],
            switch_penalty=penalties_json["switch penalty"],
            useless_comparison_penalty=penalties_json["useless comparison penalty"],
            useless_node_penalty=penalties_json["useless node penalty"],
            recurrence_penalty=penalties_json["recurrence penalty"],
            non_recurrence_penalty=penalties_json["non recurrence penalty"],
            used_feature_penalty=penalties_json["used feature penalty"],
            unused_feature_penalty=penalties_json["unused feature penalty"]
        ),
        stop_loss=strategy_json["stop loss"],
        take_profit=strategy_json["take profit"],
        max_holding_time=strategy_json["max holding time"],
    )


    return TradingConstruct(
        optimizer_params=optimizer_params,
        feature_params=FeatureParams(
            features=features
        ),
        actions_params=actions_params,
        strategy_params=strategy_params
    )
end


function describe_network(net::Network)::String

    io = IOBuffer()

    println(io, "Comparisons")
    println(io)

    comparison_indices = Dict{ComparisonNode, Int}()

    for i = eachindex(net.comparisons)
        comparison = net.comparisons[i]
        comparison_indices[comparison] = i
        println(io, "comparison ", i)
        println(io, "feature index ", comparison.feature_idx)
        println(io, "units ", comparison.units)
        println(io)
    end

    node_indices = Dict{LogicNode, Int}()

    for i = eachindex(net.nodes)
        node_indices[net.nodes[i]] = i
    end

    println(io, "Nodes")
    println(io)

    for node = net.nodes
        println(io, "node ", node_indices[node])
        if isnothing(node.in1)
            println(io, "in1: nothing")
        elseif isa(node.in1, LogicNode)
            println(io, "in1: node ", node_indices[node.in1])
        elseif isa(node.in1, ComparisonNode)
            println(io, "in1: comparison ", comparison_indices[node.in1])
        end

        if isnothing(node.in2)
            println(io, "in2: nothing")
        elseif isa(node.in2, LogicNode)
            println(io, "in2: node ", node_indices[node.in2])
        elseif isa(node.in2, ComparisonNode)
            println(io, "in2: comparison ", comparison_indices[node.in2])
        end

        println(io)
    end

    output_switches = switch_bfs(net)

    println(io, "Switch tree")
    println(io)

    switch_indices = Dict{SwitchNode, Int}()

    for i = eachindex(output_switches)
        switch_indices[output_switches[i]] = i
    end

    for switch = output_switches
        println(io, "switch ", switch_indices[switch])
        println(io, "input node: node ", node_indices[switch.input_node])

        if isa(switch.left, SwitchNode)
            println(io, "left: switch ", switch_indices[switch.left])
        else
            println(io, "left: ", switch.left)
        end

        if isa(switch.right, SwitchNode)
            println(io, "right: switch ", switch_indices[switch.right])
        else
            println(io, "right: ", switch.right)
        end

        println(io)
    end

    return String(take!(io))
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

    println(io, "best action sequence:")
    println(io, join(results.optimizer_results.best_sequence, ", "))
    println(io)
    println(io, "best network description:")
    println(io)
    println(io, describe_network(results.optimizer_results.best_network))

    return String(take!(io))
end

end