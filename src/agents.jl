
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
    context_notification_state::Int = 1
    next_agent_instructions::Union{String, Nothing} = nothing
end

@kwdef mutable struct AgentWorkspace
    agents::Vector{Agent} = []
    propsals::Vector = []
    is_voting::Bool = false
    check_votes::Bool = false
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
    instructions = read("src/instructions.txt", String)
    instructions = replace(instructions, "AGENT_NAME" => agent.name)

    agent_names = [other_agent.name for other_agent = workspace.agents]
    deleteat!(agent_names, findfirst(isequal(agent.name), agent_names))

    instructions = replace(instructions, "OTHER_AGENTS" => join(agent_names, ", ", " and "))
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
    for i = eachindex(workspace.propsals)
        if workspace.votes[i] == length(workspace.agents)
            global_commands = workspace.propsals[i]
        end
    end

    workspace.is_voting = false
    workspace.propsals = []
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
            results = run_experiment(from_json(command["construct"]))
            experiment_dict = Dict([
                "id" => length(workspace.experiments) + 1,
                "construct" => command["construct"],
                "results" => results
            ])
            push!(workspace.experiments, experiment_dict)
            workspace.global_output *= string("[NOTIFICATION] The experiment has finished running! Use the view experiment command with an id parameter of ", length(workspace.experiments), " to display the results.") 
        end
    end
end

function run_commands!(commands::Dict{String, Any}, agent::Agent, workspace::AgentWorkspace)
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
                agent.personal_output *= "id: " * experiment["id"]
                agent.personal_output *= ", out of sample sharpe ratio: " * string(experiment["results"].oos_backtest_results.sharpe_ratio)
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
            agent.personal_output *= string(workspace.experiments[idx]["construct"]) * "\n"
            agent.personal_output *= describe_experiment_results(workspace.experiments[idx]["results"]) * "\n"
        elseif command["command"] == "propose"
            if workspace.is_voting
                agent.personal_output *= "[ERROR] cannot propose duing a voting session\n"
                continue
            end
            
            if !workspace.is_voting
                workspace.is_voting = true
                workspace.global_output *= "[OUTPUT] A proposal has made been, and a voting session has been initiated! Use the vote command to vote for a proposal.\n"
            end

            workspace.global_output *= "[OUTPUT] " * agent.name * " has proposed a global command sequence!\n"
            workspace.global_output *= string("Propsal id: ", length(workspace.propsals) + 1, "\n") 
            workspace.global_output *= JSON.json(command["global commands"]) * "\n"
            push!(workspace.propsals, command["global commands"])
            push!(workspace.votes, 0)
            
        elseif command["command"] == "vote"
            if !workspace.is_voting
                agent.personal_output *= "[ERROR] voting is only allowed directly after a proposal has been made\n"
                continue
            end

            if agent.has_voted
                agent.personal_output *= "[ERROR] you can only vote once\n"
                continue
            end

            if command["proposal id"] > length(workspace.propsals)
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
            agent.next_agent_instructions = command["contents"]
            agent.personal_output *= "[OUTPUT] Instructions for next agent have been saved\n"
        end
    end
end

function prompt_agents!(workspace::AgentWorkspace)::Vector{String}
    responses::Vector{String} = []
    for agent = workspace.agents
        prompt = "GLOBAL OUTPUT\n\n"
        prompt *= workspace.global_output * "\n"
        prompt *= "PERSONAL OUPUT\n\n"
        prompt *= agent.personal_output * "\n"
        agent.personal_output = ""

        push!(responses, run_prompt!(agent, prompt))
        sleep(6.5)
    end
    workspace.global_output = ""
    return responses
end

function check_context!(workspace::AgentWorkspace, responses::Vector{String})
    for agent = workspace.agents
        if agent.context_tokens > agent.context_limit
            workspace.global_output *= "[NOTIFICATION] " * agent.name * " has reached their context limit and has been terminated. A new " * agent.name * " agent will be instantiated.\n"
            idx = findfirst(isequal(agent), workspace.agents)
            workspace.agents[idx].context = []
            workspace.agents[idx].context_tokens = 0
            workspace.agents[idx].context_notification_state = 1
            workspace.agents[idx].descendant_number += 1
            instructions = get_instructions(agent, workspace)
            if !isnothing(agent.next_agent_instructions)
                instructions *= "\nA previous " * agent.name * " agent has been active, but they have reached their context limit. Before they were terminated, they left instructions for you, the next " * agent.name * " generation.
                Here are the instructions:\n"
                instructions *= agent.next_agent_instructions
                agent.next_agent_instructions = nothing
            end
            responses[idx] = run_prompt!(agent, instructions)
        elseif agent.context_tokens > 0.8 * agent.context_limit && agent.context_notification_state == 3
            workspace.global_output *= string("[NOTIFICATION] ", agent.name, " has reached 80% of their context window\n")
            agent.personal_output *= "[NOTIFICATION] You are approaching your context token limit, so you will no longer be able to participate soon. Before you are terminated, a new " * agent.name * " agent will instantiated, but they will have no prior knowledge besides the initial instructions. As a result, you must write additional instructions for the new agent to reference. This should contain a detailed account of your thought proccesses, what have been doing, and your role in the team. The more detailed you are, the less the new model will have to repeat the thinking you already did. Use the next agent instructions command.\n"
            agent.context_notification_state = 4
        elseif agent.context_tokens > 0.5 * agent.context_limit && agent.context_notification_state == 2
            workspace.global_output *= string("[NOTIFICATION] ", agent.name, " has reached 50% of their context window\n")
            agent.context_notification_state = 3
        elseif agent.context_tokens > 0.25 * agent.context_limit && agent.context_notification_state == 1
            workspace.global_output *= string("[NOTIFICATION] ", agent.name, " has reached 25% of their context window\n")
            agent.context_notification_state = 2
        end
    end

    return responses
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
    responses::Vector{String} = []
    for agent = workspace.agents
        push!(responses, run_prompt!(agent, get_instructions(agent, workspace)))
    end

    while true
        for i = eachindex(workspace.agents)
            commands = Dict{String, Any}()
            try
                commands = JSON.parse(responses[i])
            catch e
                workspace.agents[i].personal_output *= string("[ERROR] an error occured while parsing commands JSON: ", e, ". please ensure that responses follow the JSON schema and don't have any extraneous text or unescaped special charaters\n")
            end

            try
                run_commands!(commands, workspace.agents[i], workspace)
            catch e
                workspace.agents[i].personal_output *= string("[ERROR] an error occured while running commands: ", e, "\n")
            end
        end
        
        if workspace.check_votes
            workspace.check_votes = false
            workspace.is_voting = false
            try
                run_global_commands!(workspace)
            catch e
                workspace.global_output *= string("[ERROR] an error occured while running global commands: ", e, "\n")
            end
        elseif workspace.is_voting
            workspace.check_votes = true
        end

        while true
            try
                responses = prompt_agents!(workspace)
                break
            catch e
                println(e)
                println("An error occured while prompting agents; trying again in 5 minutes")
                sleep(60)
            end
        end
        
        check_context!(workspace, responses)

        log_agents(workspace)
    end
end

function go()

    workspace = AgentWorkspace()
    create_agent!(workspace, 100000)
    create_agent!(workspace, 150000)
    create_agent!(workspace, 200000)

    run_agents!(workspace)
end


end