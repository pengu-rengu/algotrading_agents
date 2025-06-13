module BacktestModule

using JSON
using Plots
using Statistics
export backtest, BacktestResults

@kwdef struct BacktestResults
    sharpe_ratio::Float64
    entries::Int
    avg_holding_time::Float64
    is_invalid::Bool
end

function backtest(signals::Vector{Tuple{Bool, Bool}}, prices::Vector{Float64}; detailed_results::Bool = false)::Union{Float64, BacktestResults}
    
    balance = 5000.0
    enter_price = -1.0
    values::Vector{Float64} = []
    entries = 0
    enter_index = -1
    holding_times::Vector{Int} = []

    for i = eachindex(prices)[3:end]
        
        if signals[i][1] && enter_price > 0
            balance += prices[i] - enter_price
            enter_price = -1
            push!(holding_times, i - enter_index)
        elseif signals[i][2] && enter_price < 0
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

#=
function run_backtest()

    prices = get_prices()
    prices .*= 0.001

    oos_prices = prices[1:500]
    is_prices = prices[500:750]

    opt = GeneticOptimizer(
        inputs=[ContinuousFeature(0.0, 0.2, 25)],
        outputs=[DiscreteFeature(0.0, 1.0)],
        population_size=250,
        n_generations=500,
        n_elites=5,
        sequence_length=70,
        mutation_rate=0.1,
        crossover_rate=0.1,
        tournament_size=10,
        mutation_delta=0.01,
        crossover_delta=0.01,
        tournament_delta=1,
        n_gram=5,
        diversity_target=0.7,
        eval_func=net -> backtest(net, is_prices)
    )
    println(opt.outputs[1].resolution)
    optimize!(opt)
    best_net = best_network(opt)

    backtest(best_net, is_prices, plot_values=true)
    backtest(best_net, oos_prices, plot_values=true)

    print_structure(best_net)

end
=#
end