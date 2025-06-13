module StrategyModule

using ..FeaturesModule

export StrategyParams, generate_signals
export Network, NetworkParams, AbstractFeature, ContinuousFeature, SwitchNode,
DiscreteFeature, evaluate!, init_roots!, reset_state!, switch_bfs, describe_network,
get_penalty

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

abstract type AbstractNode end

mutable struct ComparisonNode <: AbstractNode
    input_feature::AbstractFeature
    input_idx::Int
    units::Int
    value::Bool
    ComparisonNode(input_feature::AbstractFeature, input_idx::Int, units::Int) =
    new(input_feature, input_idx, units, false)
end

function evaluate!(comp::ComparisonNode, inputs::Vector{Float64})::Bool
    input_value = inputs[comp.input_idx]
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
    left::Union{SwitchNode, Float64, Nothing} = nothing
    right::Union{SwitchNode, Float64, Nothing} = nothing
    value::Bool = false
end

@kwdef struct NetworkParams
    comparison_penalty::Float64
    node_penalty::Float64
    switch_penalty::Float64
    default_output::Float64
end

@kwdef mutable struct Network
    params::NetworkParams
    inputs::Vector{AbstractFeature}
    outputs::Vector{AbstractFeature}
    comparisons::Vector{ComparisonNode} = []
    nodes::Vector{LogicNode} = []
    roots::Vector{Union{SwitchNode, Nothing}} = [] 
end

function init_roots!(net::Network)
    net.roots = [nothing for _ = net.outputs]
    net.state.current_switches = [nothing for _ = net.outputs]
end

function reset_state!(net::Network)
    for comparison = net.comparisons
        comparison.value = false
    end

    for node = net.nodes
        node.value = false
    end
end

function evaluate!(net::Network, inputs::Vector{Float64})
    for comparison = net.comparisons
        evaluate!(comparison, inputs)
    end

    for node = net.nodes
        evaluate!(node)
    end

    outputs = []
    for root = net.roots
        if isnothing(root)
            push!(outputs, 0)
            continue
        end

        current::Union{SwitchNode, Float64, Nothing} = root

        while isa(current, SwitchNode)
            if current.input_node.value
                current = current.right
            else
                current = current.left
            end
        end
        if isnothing(current)
            current = net.params.default_output
        end
        push!(outputs, current)
    end

    return outputs
end



function switch_bfs(net::Network)::Vector{Vector{SwitchNode}}
    switches::Vector{Vector{SwitchNode}} = []
    
    for root = net.roots
        push!(switches, [])
        if isnothing(root)
            continue
        end

        level = [root]
        while length(level) > 0
            next_level::Vector{SwitchNode} = []
            for switch = level
                push!(switches[end], switch)
                if isa(switch.left, SwitchNode)
                    push!(next_level, switch.left)
                end

                if isa(switch.right, SwitchNode)
                    push!(next_level, switch.right)
                end
            end

            level = next_level

        end
    end

    return switches
end

function get_penalty(net::Network)::Float64
    penalty = net.params.comparison_penalty * length(net.comparisons)
    penalty += net.params.node_penalty * length(net.nodes)
    penalty += net.params.switch_penalty * length(vcat(switch_bfs(net)))
    return penalty
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
        println(io, "input index ", comparison.input_idx)
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

    all_switches = switch_bfs(net)

    for i = eachindex(all_switches)
        println(io, "Output ", i, " switch tree")
        println(io)

        output_switches = all_switches[i]

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
    end

    return String(take!(io))
end

@kwdef struct StrategyParams
    network_params::NetworkParams
end

function generate_signals(net::Network, prices::Vector{Float64}, features::Vector{InputFeature})::Vector{Tuple{Bool, Bool}}
    feature_values = [get_values(feature, prices) for feature = features]
    feature_values = [[inner[i] for inner = feature_values] for i = 1:length(feature_values[1])]
    signals::Vector{Tuple{Bool, Bool}} = []
    
    reset_state!(net)

    for i = eachindex(feature_values)
        result = evaluate!(net, feature_values[i])
        push!(signals, (result[1] > 0.5, result[2] > 0.5))
    end

    return signals
end

end