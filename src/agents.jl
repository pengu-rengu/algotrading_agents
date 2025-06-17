
module AgentsModule

using HTTP
using JSON
using ..ConstructModule

export go

api_key = ENV["GEMINI_API_KEY"]
model = "gemini-2.5-flash-preview-05-20"

@kwdef mutable struct Agent
    name::String
    context_limit::Int
    descendant_number::Int = 1
    context::Vector{Dict} = []
    personal_output::String = ""
    has_voted::Bool = false
    context_tokens::Int = 0
    context_notified::Bool = false
    next_agent_instructions::Union{String, Nothing} = nothing
end

@kwdef mutable struct AgentWorkspace
    agents::Vector{Agent} = []
    responses::Vector{String} = []
    proposals::Vector = []
    voting_state::Int = 1
    votes::Vector{Int} = []
    global_output::String = ""
    experiments::Vector{Dict} = []
    
end

function create_agent!(workspace::AgentWorkspace, context_limit::Int)
    random_name() = "GeminiFlash" * join([string(rand(0:9)) for _ = 1:3])
    agent_name = random_name()
    while agent_name in [agent.name for agent = workspace.agents]
        agent_name = random_name()
    end
    push!(workspace.agents, Agent(name=agent_name, context_limit=context_limit))
end

function get_instructions(agent::Agent, workspace::AgentWorkspace)::String
    instructions = read("src/data/instructions.txt", String)
    instructions = replace(instructions, "AGENT_NAME" => agent.name)

    agent_names = [other_agent.name for other_agent = workspace.agents]
    deleteat!(agent_names, findfirst(isequal(agent.name), agent_names))

    instructions = replace(instructions, "OTHER_AGENTS" => join(agent_names, ", ", " and "))
    instructions = replace(instructions, "FEATURES_SOURCE_CODE" => read("src/features.jl", String))
    instructions = replace(instructions, "STRATEGY_SOURCE_CODE" => read("src/strategy.jl", String))
    instructions = replace(instructions, "ACTIONS_SOURCE_CODE" => read("src/actions.jl", String))
    instructions = replace(instructions, "OPTIMIZER_SOURCE_CODE" => read("src/optimizer.jl", String))
    return instructions
end

function run_prompt!(agent::Agent, prompt::String)::String
    response = HTTP.post("https://generativelanguage.googleapis.com/v1beta/models/$(model):generateContent?key=$(api_key)",
    headers=[
        "Content-Type" => "application/json"
    ], body=JSON.json(Dict([
        "contents" => [
            agent.context...,
            Dict([
                "role" => "user",
                "parts" => [Dict([
                    "text" => prompt
                ])]
            ]),
            Dict([
                "role" => "model",
                "parts" => [Dict([
                    "text" => "{"
                ])]
            ])
        ]
    ])))
    resp_json = JSON.parse(String(response.body))
    resp_text = "{" * resp_json["candidates"][1]["content"]["parts"][1]["text"]
    resp_tokens = resp_json["usageMetadata"]["totalTokenCount"]
    agent.context_tokens = resp_tokens

    push!(agent.context, Dict([
        "role" => "user",
        "parts" => [Dict([
            "text" => prompt
        ])]
    ]))
    push!(agent.context, Dict([
        "role" => "model",
        "parts" => [Dict([
            "text" => resp_text
        ])]
    ]))

    return resp_text
end

function run_global_commands!(workspace::AgentWorkspace)

    global_commands::Union{Vector, Nothing} = nothing
    for i = eachindex(workspace.proposals)
        if workspace.votes[i] == length(workspace.agents) || (workspace.votes[i] >= length(workspace.agents) - 1 && length(workspace.proposals[i]) == 1 && workspace.proposals[i][1]["command"] == "restart")
            global_commands = workspace.proposals[i]
        end
    end

    workspace.voting_state = 1
    workspace.proposals = []
    workspace.votes = []
    for agent = workspace.agents
        agent.has_voted = false
    end

    if isnothing(global_commands)
        workspace.global_output *= "[OUTPUT] no proposal has been unanimously voted for; no global commands will be executed\n"
        return
    end

    for command = global_commands
        if command["command"] == "submit"
            results = run_experiment(construct_from_json(command["construct"]))
            experiment_dict = Dict([
                "id" => length(workspace.experiments) + 1,
                "construct" => command["construct"],
                "results" => results
            ])
            push!(workspace.experiments, experiment_dict)
            workspace.global_output *= string("[NOTIFICATION] The experiment has finished running! Use the view experiment command with an id parameter of ", length(workspace.experiments), " to display the results.") 
        elseif command["command"] == "message admin"
            open("src/data/admin_inbox.txt", "a") do file
                write(file, "\n" * command["contents"] * "\n")
            end
            workspace.global_output *= "[OUTPUT] message sent to human administrator\n"
        elseif command["command"] == "restart"
            idx = findfirst(agent -> agent.name == command["agent name"], workspace.agents)
            workspace.agents[idx].context = []
            workspace.agents[idx].context_tokens = 0
            workspace.agents[idx].context_notified = false
            workspace.agents[idx].descendant_number += 1
            instructions = get_instructions(workspace.agents[idx], workspace)
            instructions *= "\nA previous " * workspace.agents[idx].name * " agent has been active, but they were behaving problematically. Before they were terminated, they left instructions for you, the next " * workspace.agents[idx].name * " generation. Here are the instructions:\n"
            instructions *= command["instructions"]
            workspace.responses[idx] = run_prompt!(workspace.agents[idx], instructions)
            sleep(65)
            workspace.global_output *= "[NOTIFICATION] " * workspace.agents[idx].name * " has been restarted\n"
        else
            workspace.global_output *= string("[ERROR] Unrecognized global command: ", command["command"], ". Maybe it's a personal command?\n")
        end
    end
end

function run_commands!(commands::Dict{String, Any}, agent::Agent, workspace::AgentWorkspace)
    if length(commands["commands"]) == 0
        agent.personal_output *= "[ERROR] An empty command sequence is unacceptable! There is always something to be doing, whether it be discussing with your team, proposing or voting for global commands, or viewing past experiments.\n"
    end
    for command = commands["commands"]
        if command["command"] == "message"
            workspace.global_output *= string("[", agent.name, "] ", command["contents"], "\n\n")
        elseif command["command"] == "whisper"
            if length(command["recipients"]) >= length(workspace.agents) - 1
                agent.personal_output *= "[ERROR] Whispering to all the other agents is useless. Use the message command instead to send a message to all other agents.\n"
                continue
            end
            for recipient_name = command["recipients"]
                idx = findfirst(a -> a.name == recipient_name, workspace.agents)
                recipients_str = replace(join(command["recipients"], ", ", " and "), recipient_name => "you")
                workspace.agents[idx].personal_output *= string("[", agent.name, "] (whispers to ", 
                recipients_str, ") ", command["contents"], "\n")
            end
        elseif command["command"] == "list experiments"
            agent.personal_output *= "[OUTPUT]\n"
            if length(workspace.experiments) == 0
                agent.personal_output *= "no past experiments\n"
                continue
            end
            for experiment = workspace.experiments
                agent.personal_output *= string("id: ", experiment["id"], ", out of sample sharpe ratio: ", experiment["results"].oos_backtest_results.sharpe_ratio)
                agent.personal_output *= "\n"
            end
        elseif command["command"] == "view experiment"
            idx = findfirst(e -> e["id"] == command["experiment id"], workspace.experiments)
            if isnothing(idx)
                agent.personal_output *= string("[ERROR] could not find experiment with id ", command["experiment id"], "\n")
                continue
            end
            agent.personal_output *= "[OUTPUT]\n"
            agent.personal_output *= "Construct JSON:\n"
            agent.personal_output *= JSON.json(workspace.experiments[idx]["construct"]) * "\n"
            agent.personal_output *= describe_experiment_results(workspace.experiments[idx]["results"]) * "\n"
        elseif command["command"] == "propose"
            if workspace.voting_state == 3
                agent.personal_output *= "[ERROR] cannot propose during a voting session\n"
                continue
            end
            
            if workspace.voting_state == 1
                workspace.voting_state = 2
                workspace.global_output *= "[OUTPUT] A proposal has made been, and a voting session has been initiated! Use the vote command to vote for a proposal.\n"
            end

            workspace.global_output *= "[OUTPUT] " * agent.name * " has proposed a global command sequence!\n"
            workspace.global_output *= string("Proposal id: ", length(workspace.proposals) + 1, "\n") 
            workspace.global_output *= JSON.json(command["global commands"]) * "\n"
            push!(workspace.proposals, command["global commands"])
            push!(workspace.votes, 0)
            
        elseif command["command"] == "vote"
            
            if workspace.voting_state != 3
                agent.personal_output *= "[ERROR] voting is only allowed directly after a proposal has been made\n"
                continue
            end

            if agent.has_voted
                agent.personal_output *= "[ERROR] you can only vote once\n"
                continue
            end

            if command["proposal id"] > length(workspace.proposals)
                agent.personal_output *= string("[ERROR] could not find proposal with id ", command["proposal id"], "\n")
                continue
            end

            workspace.global_output *= string("[OUTPUT] ", agent.name, " has voted for proposal ", command["proposal id"], "!\n")
            workspace.votes[command["proposal id"]] += 1
            agent.has_voted = true
        elseif command["command"] == "next agent instructions"
            if agent.context_tokens < 0.8 * agent.context_limit
                agent.personal_output *= "[ERROR] Next agent instructions can only be given when you have used more than 80% of your context limit.\n"
                continue
            end
            if !isnothing(agent.next_agent_instructions)
                agent.personal_output *= "[ERROR] You have already given next agent instructions.\n"
                continue
            end
            agent.next_agent_instructions = command["contents"]
            agent.personal_output *= "[OUTPUT] Instructions for next agent have been saved\n"
        elseif command["command"] == "context used"
            agent.personal_output *= string("[OUTPUT] Context used: ", round((agent.context_tokens / agent.context_limit) * 100, digits=2), "% (", agent.context_tokens, "/", agent.context_limit, " tokens)\n")
        else
            agent.personal_output *= string("[ERROR] Unrecognized personal command: ",  command["command"], ". Maybe it's a global command?\n")
        end
    end
end

function prompt_agents!(workspace::AgentWorkspace)
    workspace.responses = []
    for agent = workspace.agents
        prompt = "GLOBAL OUTPUT\n\n"
        prompt *= workspace.global_output * "\n"
        prompt *= "PERSONAL OUPUT\n\n"
        prompt *= agent.personal_output * "\n"
        agent.personal_output = ""

        push!(workspace.responses, run_prompt!(agent, prompt))
        sleep(65)
    end
    workspace.global_output = ""
end

function check_context!(workspace::AgentWorkspace)
    for agent = workspace.agents
        if agent.context_tokens > agent.context_limit
            workspace.global_output *= "[NOTIFICATION] " * agent.name * " has reached their context limit and has been terminated. A new " * agent.name * " agent will be instantiated.\n"
            idx = findfirst(isequal(agent), workspace.agents)
            workspace.agents[idx].context = []
            workspace.agents[idx].context_tokens = 0
            workspace.agents[idx].context_notified = false
            workspace.agents[idx].descendant_number += 1
            instructions = get_instructions(agent, workspace)
            if !isnothing(agent.next_agent_instructions)
                instructions *= "\nA previous " * agent.name * " agent has been active, but they have reached their context limit. Before they were terminated, they left instructions for you, the next " * agent.name * " generation.Here are the instructions:\n"
                instructions *= agent.next_agent_instructions
                agent.next_agent_instructions = nothing
            end
            workspace.responses[idx] = run_prompt!(agent, instructions)
            sleep(65)
        elseif agent.context_tokens > 0.8 * agent.context_limit && !agent.context_notified
            workspace.global_output *= string("[NOTIFICATION] ", agent.name, " has reached 80% of their context window\n")
            agent.personal_output *= "[NOTIFICATION] You are approaching your context token. Before you are terminated, a new " * agent.name * " agent will instantiated, but they will have no prior knowledge besides the initial instructions. As a result, you must write additional instructions for the new agent to reference. This should contain a detailed account of your thought proccesses, what have been doing, and your role in the team. The more detailed you are, the less the new model will have to repeat the thinking you already did. Also remember that although you are approaching your context limit, you should continue to participate instead of waiting and doing nothing. Use the next agent instructions command.\n"
            agent.context_notified = true
        end
    end
end

function log_agents(workspace::AgentWorkspace)
    for agent = workspace.agents
        open(string("src/agent_logs/", agent.name, "-", agent.descendant_number, "_logs.txt"), "w") do file
            println(file, agent.name, "-", agent.descendant_number, " logs")
            println(file)
            for message = agent.context
                println(file, "ROLE: ", uppercase(message["role"]))
                println(file)
                println(file, "TEXT: ", message["parts"][1]["text"])
                println(file)
            end
        end
    end
end

function run_agents!(workspace::AgentWorkspace)
    workspace.responses = []
    for agent = workspace.agents
        push!(workspace.responses, run_prompt!(agent, get_instructions(agent, workspace)))
    end

    while true
        for i = eachindex(workspace.agents)
            commands = Dict{String, Any}()
            try
                commands = JSON.parse(workspace.responses[i])
            catch e
                workspace.agents[i].personal_output *= string("[ERROR] an error occured while parsing commands JSON: ", e, ". please ensure that responses follow the JSON schema and don't have any extraneous text, linebreaks, or any other unescaped special charaters\n")
            end

            try
                run_commands!(commands, workspace.agents[i], workspace)
            catch e
                workspace.agents[i].personal_output *= string("[ERROR] an error occured while running commands: ", e, "\n")
                if isa(e, KeyError)
                    workspace.agents[i].personal_output *= "[ERROR] A KeyError has occured. please ensure that your commands have all the required parameters."
                end
            end
        end
        
        if workspace.voting_state == 3
            workspace.voting_state = 1
            try
                run_global_commands!(workspace)
            catch e
                workspace.global_output *= string("[ERROR] an error occured while running global commands: ", e, "\n")
                if isa(e, KeyError)
                    workspace.global_output *= "[ERROR] A KeyError has occured. please
                    ensure that the global commands have all the required parameters."
                end
            end
        elseif workspace.voting_state == 2
            workspace.voting_state = 3
        end

        while true
            try
                prompt_agents!(workspace)
                println("successfully prompted agents")
                break
            catch e
                println(e)
                println("An error occured while prompting agents; trying again in 1 hour")
                sleep(3600)
            end
        end
        
        check_context!(workspace)

        log_agents(workspace)
    end
end

function go()

    workspace = AgentWorkspace()

    construct1_json = JSON.parse(read("src/data/experiment1.json", String))
    results1 = run_experiment(construct_from_json(construct1_json))
    println(describe_experiment_results(results1))
    experiment1_dict = Dict([
        "id" => length(workspace.experiments) + 1,
        "construct" => construct1_json,
        "results" => results1
    ])
    push!(workspace.experiments, experiment1_dict)

    """
    construct2_json = JSON.parse(read("src/data/experiment2.json", String))
    results2 = run_experiment(construct_from_json(construct2_json))
    println(describe_experiment_results(results2))
    experiment2_dict = Dict([
        "id" => length(workspace.experiments) + 1,
        "construct" => construct2_json,
        "results" => results2
    ])
    push!(workspace.experiments, experiment2_dict)
    """
    create_agent!(workspace, 100000)
    create_agent!(workspace, 150000)
    create_agent!(workspace, 200000)

    #run_agents!(workspace)
end


end