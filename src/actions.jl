module ActionsModule

using ..FeaturesModule
using ..StrategyModule

export ActionsParams, construct_network, all_actions

@kwdef struct ActionsParams
    meta_actions::Dict{String, Vector{String}}
    allow_functions::Bool
    allow_recurrence::Bool
    action_probs::Union{nothing, Vector{Float64}}
end

function all_actions(action_params::ActionsParams)
    actions = vcat("NEXT_INPUT", "NEXT_INPUT_UNIT", "NEXT_OUTPUT", "NEXT_OUTPUT_UNIT", "NEW_COMPARISON", "NEXT_COMPARISON", "NEW_NODE", "NEXT_NODE", "SELECT_NODE", "SET_IN_COMP", "SET_IN_NODE", "NEW_SWITCH", "SET_VALUE", collect(keys(action_params.meta_actions)))
    if action_params.allow_functions
        append!(actions, ["MARK_FUNCTION", "NEXT_FUNCTION", "CALL_FUNCTION"])
    end

    return actions
end

@kwdef mutable struct ActionState
    input_idx::Int = 1
    input_units::Int = 0
    output_idx::Int = 1
    output_units::Int = 0
    comp_idx::Int = 1
    node_idx::Int = 1
    selected_idx::Int = 1
    func_idx::Int = 1
    func_start_idx::Int = -1
    functions::Vector{Vector{String}} = []
    action_history::Vector{String} = []
    current_switches::Vector{Union{SwitchNode, Nothing}} = []
end

function do_action!(net::Network, action::String, state::ActionState, params::ActionsParams; append_history=true)
    if append_history
        push!(state.action_history, action)
    end
    if state.func_start_idx > 0 && action != "MARK_FUNCTION"
        return
    end

    if haskey(params.meta_actions, action)
        for sub_action = params.meta_actions[action]
            do_action!(net, sub_action, state, params, append_history=false)
        end
        return
    end

    if action == "MARK_FUNCTION"
        if state.func_start_idx < 0
            state.func_start_idx = length(state.action_history) + 1
        else
            push!(state.functions, state.action_history[state.func_start_idx:end-1])
            state.func_start_idx = -1
        end
    elseif action == "NEXT_FUNCTION"
        state.func_idx += 1
        if state.func_idx > length(state.functions)
            state.func_idx = 1
        end
    elseif action == "CALL_FUNCTION" && length(state.functions) > 0
        for func_action = state.functions[state.func_idx]
            if func_action == "CALL_FUNCTION"
                continue
            end
            do_action!(net, func_action, state, params, append_history=false)
        end
    elseif action == "NEXT_INPUT"
        state.input_idx += 1
        if state.input_idx > length(net.inputs)
            state.input_idx = 1
        end
    elseif action == "NEXT_INPUT_UNIT"
        state.input_units += 1
        if state.input_units > net.inputs[state.input_idx].resolution
            state.input_units = 0
        end
    elseif action == "NEXT_OUTPUT"
        state.output_idx += 1
        if state.output_idx > length(net.outputs)
            state.output_idx = 1
        end
    elseif action == "NEXT_OUTPUT_UNIT"
        state.output_units += 1
        if state.output_units > net.outputs[state.output_idx].resolution
            state.output_units = 0
        end
    elseif action == "NEW_COMPARISON"
        push!(net.comparisons, ComparisonNode(net.inputs[state.input_idx], 
        state.input_idx, state.input_units))
    elseif action == "NEXT_COMPARISON"
        state.comp_idx += 1
        if state.comp_idx > length(net.comparisons)
            state.comp_idx = 1
        end
    elseif action == "NEW_NODE"
        push!(net.nodes, LogicNode())
    elseif action == "NEXT_NODE"
        state.node_idx += 1
        if state.node_idx > length(net.nodes)
            state.node_idx = 1
        end
    end

    if length(net.nodes) > 0 && length(net.comparisons) > 0

        if action == "SELECT_NODE"
            state.selected_idx = state.node_idx
        elseif action == "SET_IN_COMP"
            node = net.nodes[state.node_idx]
            comparison = net.comparisons[state.comp_idx]
            if isnothing(node.in1)
                node.in1 = comparison
            elseif isnothing(node.in2)
                node.in2 = comparison
            end
        elseif action == "SET_IN_NODE"
            if !params.allow_recurrence && state.selected_idx >= state.node_idx
                return
            end
            node = net.nodes[state.node_idx]
            selected_node = net.nodes[state.selected_idx]
            if isnothing(node.in1)
                node.in1 = selected_node
            elseif isnothing(node.in2)
                node.in2 = selected_node
            end
        elseif action == "NEW_SWITCH"
            new_switch = SwitchNode(input_node=net.nodes[state.node_idx])
            current_switch = state.current_switches[state.output_idx]
            if isnothing(current_switch)
                current_switch = new_switch
                net.roots[state.output_idx] = current_switch
                state.current_switches[state.output_idx] = current_switch
            else
                if isnothing(current_switch.left)
                    new_switch.parent_switch = current_switch
                    current_switch.left = new_switch
                    current_switch = current_switch.left
                    state.current_switches[state.output_idx] = current_switch
                else
                    while !isnothing(current_switch.right) && !isnothing(current_switch.parent_switch)
                        current_switch = current_switch.parent_switch
                    end

                    if isnothing(current_switch.right)
                        new_switch.parent_switch = current_switch
                        current_switch.right = new_switch
                        current_switch = current_switch.right
                        state.current_switches[state.output_idx] = current_switch
                    end
                end
            end
        elseif action == "SET_VALUE"
            current_switch = state.current_switches[state.output_idx]
            if !isnothing(current_switch)
                if isnothing(current_switch.left)
                    current_switch.left = units_to_value(net.outputs[state.output_idx], state.output_units)
                elseif isnothing(current_switch.right)
                    current_switch.right = units_to_value(net.outputs[state.output_idx], state.output_units)
                end
            end
        end
    end
end

function construct_network(; action_sequence::Vector{String}, action_params::ActionsParams, inputs::Vector{AbstractFeature}, outputs::Vector{AbstractFeature})::Network
    net = Network(inputs=inputs, outputs=outputs)
    init_roots!(net)

    state = ActionState()
    state.current_switches = [nothing for _ = net.outputs]

    for action = action_sequence
        do_action!(net, action, state, action_params)
    end

    return net
end

end