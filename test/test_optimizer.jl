using Random

using .FeaturesModule
using .StrategyModule
using .ActionsModule
using .OptimizerModule

mutable struct PredefinedRNG <: AbstractRNG
    values::Vector{Union{Float64, Int}}
end

function mock_optimizer(rng::PredefinedRNG, eval_func::Function)
    return GeneticOptimizer(
        inputs=[ContinuousFeature(0.0, 1.0, 10)],
        outputs=[DiscreteFeature(0.0, 1.0)],
        rng=rng,
        eval_func=eval_func,
        time_limit=120.0,
        params=GeneticParams(
            population_size=100,
            n_elites=5,
            sequence_length=50,
            mutation_rate=0.1,
            crossover_rate=0.1,
            tournament_size=10,
            mutation_delta=0.01,
            crossover_delta=0.01,
            tournament_delta=1,
            mutation_rate=(0.01, 0.5),
            crossover_range=(0.01, 0.5),
            tournament_range=(2, 50),
            operator_probs=nothing,
            n_gram=5
        ),
        strategy_params=StrategyParams(
            network_penalties=NetworkPenalties(
                comparison_penalty=0.0,
                node_penalty=0.0,
                switch_penalty=0.0,
                useless_comparison_penalty=0.0,
                useless_node_penalty=0.0,
                recurrence_penalty=0.0,
                non_recurrence_penalty=0.0,
                used_feature_penalty=0.0,
                unused_feature_penalty=0.0
            )
        )
    )
end

@testset "genetic optimizer tests" begin

    @testset "test selection" begin
        opt = mock_optimizer()
    end

end