module OptimizerModule

using ..StrategyModule
using ..ActionsModule
using ..FeaturesModule
using Random
using StatsBase
using Statistics

export GeneticOptimizer, GeneticParams, GeneticResults, init_optimizer!, optimize!, best_network

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
    
    operator_probs::Union{Nothing, Vector{Float64}}

    n_gram::Int
    diversity_target::Float64
end

@kwdef mutable struct GeneticOptimizer
    
    features::Vector{AbstractFeature}
    params::GeneticParams
    action_params::ActionsParams
    time_limit::Float64
    eval_func::Function

    mutation_rate::Float64 = 0.0
    crossover_rate::Float64 = 0.0
    tournament_size::Int = 0

    rng::AbstractRNG = Random.default_rng()

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
    best_sequence::Vector{String}
end

function init_optimizer!(opt::GeneticOptimizer)
    opt.mutation_rate = opt.params.mutation_rate
    opt.crossover_rate = opt.params.crossover_rate
    opt.tournament_size = opt.params.tournament_size
    opt.population = []
    for _ = 1:opt.params.population_size
        random_sequence::Vector{String} = []
        for __ = 1:opt.params.sequence_length
            if isnothing(opt.action_params.action_probs)
                push!(random_sequence, rand(opt.rng, all_actions(opt.action_params)))
            else
                push!(random_sequence, sample(opt.rng, all_actions(opt.action_params), Weights(opt.action_params.action_probs)))
            end
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
            features=opt.features,
        )
        push!(opt.scores, opt.eval_func(net))
    end

    return maximum(opt.scores), Statistics.mean(opt.scores)
end

function select(opt::GeneticOptimizer)::Vector{String}
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

function crossover(opt::GeneticOptimizer, parent1::Vector{String}, parent2::Vector{String})::Vector{String}
    if rand(opt.rng) < opt.crossover_rate
        idx = rand(opt.rng, 1:min(length(parent1), length(parent2)))
        child = [parent1[1:idx]; parent2[idx+1:end]]
        return child
    end
    return copy(parent1)
end

function mutate(opt::GeneticOptimizer, sequence::Vector{String})::Vector{String}
    
    mutated_sequence = copy(sequence)
    
    for _ = sequence
        if rand(opt.rng) > opt.mutation_rate
            continue
        end

        idx = rand(opt.rng, 1:length(mutated_sequence))
        if isnothing(opt.action_params.action_probs)
            mutated_sequence[idx] = rand(opt.rng, all_actions(opt.action_params))
        else
            mutated_sequence[idx] = sample(opt.rng, all_actions(opt.action_params), Weights(opt.action_params.action_probs))
        end
    end
    return mutated_sequence
end

function calculate_diversity(opt::GeneticOptimizer, population::Vector{Vector{String}})::Float64
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
    operator = ""
    if isnothing(opt.params.operator_probs)
        operator = rand(opt.rng, ["mutation", "crossover", "tournament"])
    else
        operator = sample(opt.rng, ["mutation", "crossover", "tournament"], Weights(opt.params.operator_probs))
    end
    
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

function get_elites(opt::GeneticOptimizer)::Vector{Vector{String}}
    elites = Vector{Vector{String}}()
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

function get_new_population(opt::GeneticOptimizer)::Vector{Vector{String}}
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
        best_network=best_network(opt),
        best_sequence=opt.population[findmax(opt.scores)[2]]
    )
end

function best_network(opt::GeneticOptimizer)::Network
    idx = findmax(opt.scores)[2]
    sequence = opt.population[idx]

    net = construct_network(
        action_sequence=sequence,
        action_params=opt.action_params,
        features=opt.features
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

end
