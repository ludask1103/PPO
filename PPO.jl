using Flux, Plots, ReinforcementLearning, Statistics, Random, ProgressBars, StableRNGs, ChainRules
using StatsBase: entropy, wsample
theme(:ggplot2)

rng = StableRNG(1)

mutable struct Trajectory
    num_environments ::Int64
    time_steps       ::Int64
    num_states       ::Int64
    num_actions      ::Int64
    states           ::Array{Float64}
    actions          ::Matrix{Int64}
    probs            ::Array{Float64}
    reward           ::Matrix{Float64}
    value            ::Matrix{Float64}
    done             ::Matrix{Bool}
    advantage        ::Matrix{Float64}
    returns          ::Matrix{Float64}
    
    function Trajectory(num_environments::Int64,
                        time_steps::Int64, 
                        num_states::Int64, 
                        num_actions::Int64)

        states    = zeros(Float64,num_states, time_steps, num_environments)
        actions   = zeros(Int64, time_steps, num_environments)
        probs     = zeros(Float64, num_actions, time_steps, num_environments)
        reward    = zeros(Float64, time_steps, num_environments)
        value     = zeros(Float64, time_steps, num_environments)
        done      = zeros(Bool, time_steps, num_environments)
        advantage = zeros(Float64, time_steps, num_environments)
        returns   = zeros(Float64, time_steps, num_environments)

        new(num_environments,time_steps, num_states, num_actions, states, actions, probs, reward, value, done, advantage, returns)
    end
end

mutable struct Agent
    actor  ::Chain
    critic ::Chain
end

mutable struct Worker   
    trajectory       ::Trajectory
    agent            ::Agent
    environments     ::Vector{<:AbstractEnv}
    time_steps       ::Int64
    num_environments ::Int64

    function Worker(num_environments::Int64,
                    time_steps::Int64,
                    environments::Vector{<:AbstractEnv},
                    actor::Chain,
                    critic::Chain)

        num_states  = length(state(environments[1]))
        num_actions = length(action_space(environments[1]))
        trajectory  = Trajectory(num_environments,time_steps,num_states,num_actions)

        agent = Agent(actor,critic)
        new(trajectory,agent,environments,time_steps,num_environments)
    end
end

mutable struct Logger    
    Size          ::Int64
    index         ::Int64
    L_entropy     ::Vector{Float64}
    L_value       ::Vector{Float64}
    L_clip        ::Vector{Float64}
    L_total       ::Vector{Float64}     
    kl_divergence ::Vector{Float64}

    function Logger(Size::Int64)
        index         = 1
        L_entropy     = zeros(Float64,Size)
        L_value       = zeros(Float64,Size)
        L_clip        = zeros(Float64,Size)
        L_total       = zeros(Float64,Size)
        kl_divergence = zeros(Float64,Size)  
        new(Size,index,L_entropy,L_value,L_clip,L_total,kl_divergence)
    end
end

function run_policy(
    worker::Worker
    )::Worker

    for i in 1:worker.time_steps
        for j in 1:worker.num_environments
            if is_terminated(worker.environments[j])
                reset!(worker.environments[j])
            end
        end

        states = reduce(hcat,state.(worker.environments))

        probs = worker.agent.actor(states)
        probs = softmax(probs)

        actions = wsample.(rng,action_space.(worker.environments),eachcol(probs))

        for j in 1:worker.num_environments
            act!(worker.environments[j],actions[j]) #wish this could have been nicer
        end

        rewards = reward.(worker.environments)
        values = worker.agent.critic(states)

        worker.trajectory.states[:,i,:] = states
        worker.trajectory.probs[:,i,:]  = log.(probs)
        worker.trajectory.actions[i,:]  = actions
        worker.trajectory.reward[i,:]   = rewards
        worker.trajectory.value[i,:]    = values
        worker.trajectory.done[i,:]     = is_terminated.(worker.environments)
    end

    return worker
end

function calculate_advantage_estimate(
    worker::Worker,               
    γ::Float64=0.99,
    λ::Float64=0.95
    )::Worker
    
    A_prime = zeros(worker.num_environments)

    T = worker.time_steps
    for i in T-1:-1:1
        mask = 1.0 .- worker.trajectory.done[i,:] #need to ignore next value if it is part of a new session 
        A_prime = A_prime .* mask  

        δ = worker.trajectory.reward[i,:] .+ γ*worker.trajectory.value[i+1,:] .* mask .- worker.trajectory.value[i,:]

        A = δ .+ γ*λ*A_prime

        worker.trajectory.advantage[i,:] = A

        A_prime = A
    end

    return worker
end

function calculate_discounted_rewards(
    worker::Worker,
    γ::Float64=0.99
    )::Worker

    T = worker.trajectory.time_steps
    return_sum = zeros(worker.num_environments)
    for i in T:-1:1

        return_sum = return_sum .* (1 .- worker.trajectory.done[i,:])

        return_sum = worker.trajectory.reward[i,:] .+ γ*return_sum

        worker.trajectory.returns[i,:] = return_sum
    end
    
    return worker
end

function create_mini_batches(
    sample_length::Int64, 
    batch_size::Int64
    )::Vector{Vector{Int64}}

    ind = shuffle(rng,collect(1:sample_length))
    batch_ind = []
    
    while true
        if length(ind) >= batch_size
            push!(batch_ind, splice!(ind,1:batch_size))
        else
            push!(batch_ind, ind)
            break
        end
    end

    return batch_ind[1:end-1]
end

function update_params(
    optimiser::Flux.Optimise.AbstractOptimiser,
    worker::Worker,
    n_epochs::Int64,
    batch_size::Int64,
    c_1::Float64,
    c_2::Float64,
    β::Float64,
    ϵ::Float64,
    logger::Logger
    )::Worker

    NT = worker.time_steps*worker.num_environments

    batch_advantage = flatten_batch(worker.trajectory.advantage)
    batch_returns   = flatten_batch(worker.trajectory.returns) 
    batch_states    = flatten_batch(worker.trajectory.states) 
    batch_actions   = flatten_batch(worker.trajectory.actions) 
    batch_probs     = flatten_batch(worker.trajectory.probs)
    batch_values    = flatten_batch(worker.trajectory.value) 

    for k in 1:n_epochs
        batch_ind = create_mini_batches(NT, batch_size)
        
        for ind in batch_ind

            advantage = batch_advantage[ind]
            returns = batch_values[ind] .+ advantage
            advantage = (advantage .- mean(advantage))./(std(advantage) + 1e-8)

            #returns = batch_returns[ind] 
            returns = (returns .- mean(returns))./(std(returns) + 1e-8)
            
            states   = batch_states[:,ind]
            actions  = batch_actions[ind]
            prob_old = batch_probs[:,ind]

            prob_old_a = prob_old[actions,:]

            ps = Flux.params(worker.agent.actor,worker.agent.critic)
            gs = gradient(ps) do
                prob = worker.agent.actor(states)

                prob = softmax(prob)

                prob_log = log.(prob)

                prob_a = prob_log[actions,:]
                
                ratio = exp.(prob_a .- prob_old_a)

                surr_1 = ratio.*advantage
                surr_2 = clamp.(ratio,1-ϵ,1+ϵ).*advantage

                L_clip = mean(min.(surr_1, surr_2))

                L_entropy = mean(entropy.(eachcol(prob)))
                kl        = Flux.kldivergence(prob,exp.(prob_old))

                value = worker.agent.critic(states)[1,:]
                L_vf  = Flux.Losses.huber_loss(value,returns)

                ChainRules.ignore_derivatives() do
                    logger.L_value[logger.index]       = L_vf
                    logger.L_entropy[logger.index]     = L_entropy
                    logger.L_clip[logger.index]        = L_clip
                    logger.kl_divergence[logger.index] = kl
                    logger.L_total[logger.index]       += (L_clip + c_2*L_entropy - β*kl - c_1*L_vf)
                    logger.index                       += 1
                end

                -(L_clip + c_2*L_entropy - β*kl - c_1*L_vf)
            end

            clip_by_global_norm!(gs, ps, 0.5f0)
            Flux.update!(optimiser, ps, gs)
        end
    end

    return worker
end

function test_actor(
    actor::Chain,
    environment::AbstractEnv
    )::Float64
    rewards = []
    for _ in 1:20 
        reset!(environment)
        tmp = 0   
        while !is_terminated(environment)
            s = state(environment)
            
            probs = softmax(actor(s))
            action = wsample(rng,action_space(environment),probs)

            act!(environment,action)

            tmp += reward(environment)
        end
        push!(rewards, tmp)
    end
    return mean(rewards)
end

function main()

    lr               = 3e-4
    iterations       = 1000
    time_steps       = 512
    num_environments = 10
    n_epochs         = 10
    batch_size       = 64
    c_1              = 0.5
    c_2              = 0.01
    β                = 0.0
    ϵ                = 0.2 

    network_size = 64

    environment = CartPoleEnv(rng=rng)
    num_actions = length(action_space(environment))
    num_states  = length(state(environment))

    shared_layer = Dense(num_states => network_size, relu; init=orthogonal(rng))

    actor = Chain(shared_layer,
                    Dense(network_size => network_size,relu; init=orthogonal(rng)),
                    Dense(network_size => num_actions; init=orthogonal(rng)))

    critic = Chain(shared_layer,
                    Dense(network_size => network_size,relu; init=orthogonal(rng)),
                    Dense(network_size => 1; init=orthogonal(rng)))

    optimiser  = Adam(lr)
    environments = [CartPoleEnv(rng = StableRNG(hash(1+i))) for i in 1:num_environments]
    worker = Worker(num_environments,time_steps,environments,actor,critic)

    logger = Logger(Int(iterations*time_steps*num_environments*n_epochs/batch_size))
    
    r_baseline = []
    span_baseline = ([],[])
    for i in 1:iterations
        rew = test_actor(actor,environment)
        push!(r_baseline, rew)
    end

    r = []
    span = ([],[])
    iter = ProgressBar(1:iterations)

    for i in iter

        worker = worker |> run_policy |> calculate_advantage_estimate |> calculate_discounted_rewards

        worker = update_params(optimiser,worker,n_epochs,batch_size,c_1,c_2,β,ϵ,logger)

        rew = test_actor(worker.agent.actor, environment)
        
        push!(r, rew)

        set_postfix(iter,(Reward=string(rew)))

    end

    p1 = plot(range(1,iterations), r; label="Trained agent", title="Rewards")
    plot!(range(1,iterations), r_baseline; label="Baseline",legend=:topleft)
    p2 = plot(1:logger.Size, logger.L_entropy; title="Entropy", legend=false)
    p3 = plot(1:logger.Size, logger.kl_divergence; title="KL Divergence", legend=false)    
    p4 = plot(1:logger.Size, logger.L_clip; title="CLIP Loss", legend=false,xaxis=false)
    p5 = plot(1:logger.Size, logger.L_value; title="Value Loss", legend=false,xaxis=false)
    p6 = plot(1:logger.Size, logger.L_total; title="Total Loss", legend=false,xaxis=false)
    custom_layout = @layout [
    a                       
    b                       
    c                      
    d{0.33w} e{0.33w} f{0.33w}
    ]
    plot(p1,p2,p3,p4,p5,p6,layout=custom_layout,size=(800,1000),titleloc=:left)

end

main()