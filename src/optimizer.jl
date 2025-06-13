module OptimizerModule

using ..StrategyModule
using ..ActionsModule
using Statistics
using Random
using Plots

export GeneticOptimizer, GeneticParams, GeneticResults, init_optimizer!, optimize!, 
plot_history, best_network, run_optimizer

@kwdef struct GeneticParams
    population_size::Int
    n_elites::Int
    sequence_length::Int

    mutation_rate::Float64
    crossover_rate::Float64
    tournament_size::Int

    mutation_delta::Float64
    crossover_delta::Float64
    tournament_delta::Int

    mutation_range::Tuple{Float64, Float64}
    crossover_range::Tuple{Float64, Float64}
    tournament_range::Tuple{Float64, Float64}

    n_gram::Int
    diversity_target::Float64
end

@kwdef mutable struct GeneticOptimizer
    inputs::Vector{AbstractFeature}
    outputs::Vector{AbstractFeature}
    params::GeneticParams
    network_params::NetworkParams
    action_params::ActionParams
    time_limit::Float64
    eval_func::Function

    mutation_rate::Float64 = 0.0
    crossover_rate::Float64 = 0.0
    tournament_size::Int = 0

    population::Vector{Vector{String}} = []
    scores::Vector{Float64} = []

    mean_score_history::Vector{Float64} = []
    best_score_history::Vector{Float64} = []
    logs::Vector{String} = []
end

@kwdef struct GeneticResults
    best_score::Float64
    generations::Int
    inflection_points::Vector{Tuple{Int, Float64}}
    best_network::Network
end

function init_optimizer!(opt::GeneticOptimizer)
    opt.mutation_rate = opt.params.mutation_rate
    opt.crossover_rate = opt.params.crossover_rate
    opt.tournament_size = opt.params.tournament_size
    opt.population = []
    for _ = 1:opt.params.population_size
        random_sequence::Vector{Action} = []
        for __ = 1:opt.params.sequence_length
            push!(random_sequence, rand(all_actions(opt.action_params)))
        end
        push!(opt.population, random_sequence)
    end
end

function evaluate_scores!(opt::GeneticOptimizer)::Tuple{Float64, Float64}
    opt.scores::Vector{Float64} = []
    for i = 1:opt.params.population_size
        sequence = opt.population[i]
        net = construct_network(
            action_sequence=sequence,
            action_params=opt.action_params,
            network_params=opt.network_params,
            inputs=opt.inputs,
            outputs=opt.outputs
        )
        push!(opt.scores, opt.eval_func(net) - get_penalty(net))
    end

    return maximum(opt.scores), mean(opt.scores)
end

function select(opt::GeneticOptimizer)::Vector{Action}
    tournament = shuffle(collect(1:opt.params.population_size))[1:opt.tournament_size]
    max_idx = tournament[1]
    max_score = opt.scores[tournament[1]]
    for idx = tournament
        if opt.scores[idx] > max_score
            max_score = opt.scores[idx]
            max_idx = idx
        end
    end
    return opt.population[max_idx]
end

function crossover(opt::GeneticOptimizer, parent1::Vector{Action}, parent2::Vector{Action})::Vector{Action}
    if rand() < opt.crossover_rate
        idx = rand(1:min(length(parent1), length(parent2)))
        child = [parent1[1:idx]; parent2[idx+1:end]]
        return child
    end
    return copy(parent1)
end

function mutate(opt::GeneticOptimizer, sequence::Vector{Action})::Vector{Action}
    
    mutated_sequence = copy(sequence)
    
    for _ = sequence
        
        if rand() > opt.mutation_rate
            continue
        end

        idx = rand(1:length(mutated_sequence))
        mutated_sequence[idx] = rand(all_actions(opt.action_params))
    end
    return mutated_sequence
end

function calculate_diversity(opt::GeneticOptimizer, population::Vector{Vector{Action}})::Float64
    unique_n_grams = Set{Tuple}()

    for sequence = population
        for i = 1:(length(sequence) - opt.params.n_gram)
            push!(unique_n_grams, Tuple(sequence[i:i+opt.params.n_gram]))
        end
    end

    total_n_grams = sum([length(sequence) - opt.params.n_gram + 1 for sequence = population])
    return length(unique_n_grams) / total_n_grams
end

function adjust!(opt::GeneticOptimizer, diversity::Float64)
    operator = rand(["mutation", "crossover", "tournament"])
    direction = diversity < opt.params.diversity_target ? 1 : -1

    if operator == "mutation"
        opt.mutation_rate += direction * opt.params.mutation_delta
    elseif operator == "crossover"
        opt.crossover_rate += direction * opt.params.crossover_delta
    else
        opt.tournament_size -= direction * opt.params.tournament_delta
    end
    
    if direction == 1
        if opt.mutation_rate > opt.params.mutation_range[2]
            opt.mutation_rate = opt.params.mutation_range[2]
        elseif opt.crossover_rate > opt.params.crossover_range[2]
            opt.crossover_rate = opt.params.crossover_range[2]
        elseif opt.tournament_size < opt.params.tournament_range[1]
            opt.tournament_size = opt.params.tournament_range[1]
        end
    else
        if opt.mutation_rate < opt.params.mutation_range[1]
            opt.mutation_rate = opt.params.mutation_range[1]
        elseif opt.crossover_rate < opt.params.crossover_range[1]
            opt.crossover_rate = opt.params.crossover_range[1]
        elseif opt.tournament_size > opt.params.tournament_range[2]
            opt.tournament_size = opt.params.tournament_range[2]
        end
    end
end

function get_elites(opt::GeneticOptimizer)::Vector{Vector{Action}}
    elites = Vector{Vector{Action}}()
    indices = reverse(sortperm(opt.scores))
    for idx = indices
        sequence = opt.population[idx]
        
        push!(elites, sequence)
        if length(elites) >= opt.params.n_elites
            break
        end
    end
    return elites
end

function get_new_population(opt::GeneticOptimizer)::Vector{Vector{Action}}
    new_population = get_elites(opt)
    while length(new_population) < opt.params.population_size
        parent1 = select(opt)
        parent2 = select(opt)

        child = crossover(opt, parent1, parent2)
        child = mutate(opt, child)

        push!(new_population, child)
    end
    return new_population
end

function optimize!(opt::GeneticOptimizer)::GeneticResults
    start_time = time()
    elapsed_time = 0.0
    generation = 0

    prev_best = -999999999.0
    inflection_points::Vector{Tuple{Int, Float64}} = []

    while elapsed_time < opt.time_limit
        generation += 1
        best_score, mean_score = evaluate_scores!(opt)
        diversity = calculate_diversity(opt, opt.population)
        adjust!(opt, diversity)
        opt.population = get_new_population(opt)

        if best_score > prev_best
            push!(inflection_points, (generation, best_score))
            prev_best = best_score
        end
        push!(opt.mean_score_history, mean_score)
        push!(opt.best_score_history, best_score)

        println("=======================")
        println("elapsed_time ", round(elapsed_time, digits=2))
        println("generation ", generation)
        println("best score ", best_score)
        println("mean score ", mean_score)
        println("diversity ", diversity)
        println("mutation rate ", opt.mutation_rate)
        println("crossover rate ", opt.crossover_rate)
        println("tournament size ", opt.tournament_size)
        elapsed_time = time() - start_time
    end

    return GeneticResults(
        best_score=opt.best_score_history[end],
        generations=generation,
        inflection_points=inflection_points,
        best_network=best_network(opt)
    )
end

function best_network(opt::GeneticOptimizer)::Network
    idx = findmax(opt.scores)[2]
    sequence = opt.population[idx]

    net = construct_network(
        action_sequence=sequence,
        action_params=opt.action_params,
        network_params=opt.network_params,
        inputs=opt.inputs,
        outputs=opt.outputs
    )
    return net
end

function save(opt::GeneticOptimizer, file_path::String)
    text = ""
    actions = collect(instances(Action))
    for i = eachindex(opt.population)
        for k = eachindex(opt.population[i])
            text *= string(findfirst(a -> a==opt.population[i][k], actions))
            if k < length(opt.population[i])
                text *= " "
            end
        end
        text *= "," * string(opt.scores[i])
        text *= "\n"
    end
    
    open(file_path, "w") do file
        write(file, text)
    end
end


function plot_history(opt::GeneticOptimizer)
    p = plot([opt.best_score_history opt.mean_score_history])
    display(p)
    readline()
end

function plot_preds(opt::GeneticOptimizer)
    idx = findmax(opt.scores)[2]
    sequence = opt.population[idx]

    net = Network(inputs=opt.inputs, outputs=opt.outputs, state=NetworkState())
    init_roots!(net)
    for action = sequence
        do_action!(net, action)
    end

    x_values = collect(LinRange(0.0, 2pi, 10))
    y_values = sin.(x_values)

    preds = Vector{Float64}()

    for i = eachindex(x_values)
        x_norm = x_values[i] / (2pi)
        push!(preds, evaluate!(net, [x_norm])[1])
    end

    p = plot(x_values, [y_values preds])
    display(p)
    readline()
end


function criterion(net::Network)::Float64
    x_values = collect(LinRange(0.0, 2pi, 10))
    y_values = sin.(x_values)
    
    error = 0.0

    for i = eachindex(x_values)
        x_norm = x_values[i] / (2pi)
        pred = evaluate!(net, [x_norm])[1]
        error += (pred - y_values[i]) ^ 2
    end

    return error / length(x_values)
end

function run_optimizer()

    opt = GeneticOptimizer(
        inputs=[ContinuousFeature(0.0, 1.0, 10)],
        outputs=[ContinuousFeature(-1.0, 1.0, 10)],
        params=GeneticParams(
            population_size=1000,
            n_generations=1000,
            n_elites=5,
            sequence_length=70,
            mutation_rate=0.1,
            crossover_rate=0.1,
            tournament_size=10,
            mutation_delta=0.01,
            crossover_delta=0.01,
            tournament_delta=1,
            mutation_range=(0.01,0.5),
            crossover_range=(0.01, 0.5),
            tournament_range=(2, 100),
            n_gram=5,
            diversity_target=0.7,
        ),
        eval_func=criterion
    )
    println(opt.params)

    optimize!(opt)
    plot_history(opt)
    plot_preds(opt)
    #save(opt, joinpath(@__DIR__, "dataset.txt"))

end

end
