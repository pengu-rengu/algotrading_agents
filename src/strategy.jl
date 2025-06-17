module StrategyModule

using ..FeaturesModule
using Statistics

export StrategyParams, generate_signals
export backtest, BacktestResults
export Network, NetworkPenalties, SwitchNode, ComparisonNode, LogicNode, evaluate!, init_roots!, reset_state!, switch_bfs, get_penalty

abstract type AbstractNode end

mutable struct ComparisonNode <: AbstractNode
    input_feature::AbstractFeature
    feature_idx::Int
    units::Int
    value::Bool
    ComparisonNode(input_feature::AbstractFeature, feature_idx::Int, units::Int) =
    new(input_feature, feature_idx, units, false)
end

function evaluate!(comp::ComparisonNode, inputs::Vector{Float64})::Bool
    input_value = inputs[comp.feature_idx]
    if isa(comp.input_feature, ContinuousFeature)
        lower = units_to_value(comp.input_feature, comp.units)
        upper = units_to_value(comp.input_feature, comp.units + 1)
        comp.value = lower <= input_value < upper
    else
        comp.value = input_value == units_to_value(comp.input_feature, comp.units)
    end
end

@kwdef mutable struct LogicNode <: AbstractNode
    in1::Union{AbstractNode, Nothing} = nothing
    in2::Union{AbstractNode, Nothing} = nothing
    value::Bool = false
end

function evaluate!(logic::LogicNode)
    value1 = isnothing(logic.in1) ? false : logic.in1.value
    value2 = isnothing(logic.in2) ? false : logic.in2.value
    logic.value = !(value1 && value2)
end

@kwdef mutable struct SwitchNode
    input_node::LogicNode
    parent_switch::Union{SwitchNode, Nothing} = nothing
    left::Union{SwitchNode, Bool, Nothing} = nothing
    right::Union{SwitchNode, Bool, Nothing} = nothing
    value::Bool = false
end

@kwdef mutable struct Network
    features::Vector{AbstractFeature}
    comparisons::Vector{ComparisonNode} = []
    nodes::Vector{LogicNode} = []
    output_root::Union{SwitchNode, Nothing} = nothing
end

function reset_state!(net::Network)
    for comparison = net.comparisons
        comparison.value = false
    end

    for node = net.nodes
        node.value = false
    end
end

function evaluate!(net::Network, inputs::Vector{Float64})::Bool
    for comparison = net.comparisons
        evaluate!(comparison, inputs)
    end

    for node = net.nodes
        evaluate!(node)
    end

    if isnothing(net.output_root)
        return false
    end

    current::Union{SwitchNode, Bool, Nothing} = net.output_root

    while isa(current, SwitchNode)
        if current.input_node.value
            current = current.right
        else
            current = current.left
        end
    end

    if isnothing(current)
        return false
    end

    return current
end

function switch_bfs(net::Network)::Vector{SwitchNode}
    if isnothing(net.output_root)
        return []
    end

    switches::Vector{SwitchNode} = []
    
    level = [net.output_root]
    while length(level) > 0
        next_level::Vector{SwitchNode} = []
        for switch = level
            push!(switches, switch)
            if isa(switch.left, SwitchNode)
                push!(next_level, switch.left)
            end

            if isa(switch.right, SwitchNode)
                push!(next_level, switch.right)
            end
        end

        level = next_level

    end

    return switches
end

@kwdef struct NetworkPenalties
    comparison_penalty::Float64
    node_penalty::Float64
    switch_penalty::Float64
    useless_comparison_penalty::Float64
    useless_node_penalty::Float64
    recurrence_penalty::Float64
    non_recurrence_penalty::Float64
    used_feature_penalty::Float64
    unused_feature_penalty::Float64
end

function get_penalty(net::Network, penalties::NetworkPenalties)::Float64
    penalty = penalties.comparison_penalty * length(net.comparisons)
    penalty += penalties.node_penalty * length(net.nodes)
    penalty += penalties.switch_penalty * length(switch_bfs(net))

    if penalties.useless_comparison_penalty > 0.0
        unused_comparisons = copy(net.comparisons)
        for node = net.nodes
            idx = nothing
            if isa(node.in1, ComparisonNode)
                idx = findfirst(isequal(node.in1), unused_comparisons)
            elseif isa(node.in2, ComparisonNode)
                idx = findfirst(isequal(node.in2), unused_comparisons)
            else
                continue
            end
            
            if !isnothing(idx)
                deleteat!(unused_comparisons, idx)
            end
        end
    end

    if penalties.recurrence_penalty > 0.0 || penalties.non_recurrence_penalty > 0.0
        for i = eachindex(net.nodes)
            node = net.nodes[i]
            idx1 = isnothing(node.in1) || isa(node.in1, ComparisonNode) ? nothing : findfirst(isequal(node.in1), net.nodes)
            idx2 = isnothing(node.in2) || isa(node.in2, ComparisonNode) ? nothing :
            findfirst(isequal(node.in2), net.nodes)

            if (!isnothing(idx1) && idx1 >= i) || (!isnothing(idx2) && idx2 >= i)
                penalty += penalties.recurrence_penalty
            else
                penalty += penalties.non_recurrence_penalty
            end
        end
    end

    if penalties.used_feature_penalty > 0.0 || penalties.used_feature_penalty > 0.0
        unused_features = copy(net.features)
        for comparison = net.comparisons
            idx = findfirst(isequal(comparison.input_feature), unused_features)
            if !isnothing(idx)
                deleteat!(unused_features, idx)
            end
        end
        penalty += penalties.unused_feature_penalty * length(unused_features)
        penalty += penalties.used_feature_penalty * (length(net.features) - length(unused_features))
    end

    return penalty
end

@kwdef struct StrategyParams
    network_penalties::NetworkPenalties
    stop_loss::Union{Nothing, Float64}
    take_profit::Union{Nothing, Float64}
    max_holding_time::Union{Nothing, Int}
end

function generate_signals(net::Network, prices::Vector{Float64}, features::Vector{InputFeature})::Vector{Bool}
    feature_values = [get_values(feature, prices) for feature = features]
    feature_values = [[inner[i] for inner = feature_values] for i = 1:length(feature_values[1])]
    signals::Vector{Bool} = []
    
    reset_state!(net)

    for i = eachindex(feature_values)
        push!(signals, evaluate!(net, feature_values[i]))
    end

    return signals
end


@kwdef struct BacktestResults
    sharpe_ratio::Float64
    entries::Int
    avg_holding_time::Float64
    is_invalid::Bool
end

function backtest(signals::Vector{Bool}, prices::Vector{Float64}, strategy_params::StrategyParams; detailed_results::Bool = false)::Union{Float64, BacktestResults}
    
    balance = 5000.0
    enter_price = -1.0
    values::Vector{Float64} = []
    entries = 0
    enter_index = -1
    holding_times::Vector{Int} = []

    for i = eachindex(prices)[3:end]
        
        max_holding_time = isnothing(strategy_params.max_holding_time) ? false : i - enter_index > strategy_params.max_holding_time
        take_profit = isnothing(strategy_params.take_profit) ? false : prices[i] / enter_price > 1 + strategy_params.take_profit
        stop_loss = isnothing(strategy_params.stop_loss) ? false : prices[i] / enter_price < 1 - strategy_params.stop_loss

        if enter_price > 0 && (signals[i] || max_holding_time || take_profit || stop_loss)
            balance += prices[i] - enter_price
            enter_price = -1
            push!(holding_times, i - enter_index)
        elseif signals[i] && enter_price < 0
            enter_price = prices[i]
            entries += 1
            enter_index = i
        end

        if mod(i, 10) == 0
            if enter_price > 0
                push!(values, balance + (prices[i] - enter_price))
            else
                push!(values, balance)
            end
        end
    end

    if any(values .< 0.0) || entries < 2
        if detailed_results
            return BacktestResults(
                sharpe_ratio=0.0,
                entries=entries,
                avg_holding_time=mean(holding_times),
                is_invalid=true
            )
        end
        return 0.0
    end
    
    log_returns::Vector{Float64} = []
    
    for i = eachindex(values)[2:end]
        push!(log_returns, log(values[i] / values[i - 1]))
    end

    if length(log_returns) < 2 || std(log_returns) == 0
        return 0.0
    end
        
    sharpe_ratio = mean(log_returns) / std(log_returns)

    if detailed_results
        return BacktestResults(
            sharpe_ratio=sharpe_ratio,
            entries=entries,
            avg_holding_time=mean(holding_times),
            is_invalid=false
        )
    end

    return sharpe_ratio
end

end