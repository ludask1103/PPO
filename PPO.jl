using Flux, Plots, ReinforcementLearning, Statistics, Random, ProgressBars, StableRNGs, ChainRules
using StatsBase: entropy, wsample
theme(:ggplot2)

rng = StableRNG(1)

mutable struct Trajectory
    time_steps  ::Int64
    num_states  ::Int64
    num_actions ::Int64
    state       ::Matrix{Float64}
    action      ::Vector{Int64}
    prob        ::Matrix{Float64}
    reward      ::Vector{Float64}
    value       ::Vector{Float64}
    done        ::Vector{Bool}
    advantage   ::Vector{Float64}
    
    function Trajectory(time_steps::Int64, num_states::Int64, num_actions::Int64)
        state     = zeros(Float64, time_steps, num_states)
        action    = zeros(Int64, time_steps)
        prob      = zeros(Float64, time_steps, num_actions)
        reward    = zeros(Float64, time_steps)
        value     = zeros(Float64, time_steps)
        done      = zeros(Bool,time_steps)
        advantage = zeros(Float64, time_steps)
        new(time_steps, num_states, num_actions, state, action, prob, reward, value, done, advantage)
    end
end

mutable struct Agent
    actor  ::Chain
    critic ::Chain
end

mutable struct Worker   
    trajectory  ::Trajectory
    agent       ::Agent
    environment ::AbstractEnv
    time_steps  ::Int64

    function Worker(time_steps::Int64,environment::AbstractEnv,actor::Chain,critic::Chain)
        num_states  = length(state_space(environment))
        num_actions = length(action_space(environment))
        trajectory  = Trajectory(time_steps,num_states,num_actions)

        agent = Agent(actor,critic)
        new(trajectory,agent,environment,time_steps)
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
        if is_terminated(worker.environment)
            reset!(worker.environment)
        end

        s = state(worker.environment)

        probs = worker.agent.actor(s)
        probs = logsoftmax(probs)

        a = wsample(rng,collect(1:length(probs)),exp.(probs))
        worker.environment(a)

        r = reward(worker.environment)
        v = worker.agent.critic(s)[1]
        
        probs = Matrix(transpose(probs)) 
        s = Matrix(transpose(s))

        worker.trajectory.state[i,:]  = s
        worker.trajectory.prob[i,:]   = probs
        worker.trajectory.action[i]   = a
        worker.trajectory.reward[i]   = r
        worker.trajectory.value[i]    = v
        worker.trajectory.done[i]     = is_terminated(worker.environment)
    end

    return worker
end

function calculate_advantage_estimate(
    worker::Worker,               
    γ::Float64=0.99,
    λ::Float64=0.95
    )::Worker
    
    A_prime = 0

    T = length(worker.trajectory.action)
    for i in T-1:-1:1
        mask = 1.0 - worker.trajectory.done[i] #need to ignore next value if it is part of a new session 
        A_prime *= mask  
        δ = worker.trajectory.reward[i] + γ*worker.trajectory.value[i+1]*mask - worker.trajectory.value[i]

        A = δ + γ*λ*A_prime

        worker.trajectory.advantage[i] = A

        A_prime = A
    end

    return worker
end

function collect_trajectories(
    workers::Vector{Worker}
    )::Trajectory

    n_envs      = length(workers)
    time_steps  = workers[1].time_steps
    num_states  = length(state_space(workers[1].environment))
    num_actions = length(action_space(workers[1].environment)) 

    traj = Trajectory(time_steps*n_envs,num_states,num_actions)

    for (i,w) in enumerate(workers)
        t = w.trajectory
        traj.action[1+(i-1)*time_steps:i*time_steps]    = t.action 
        traj.advantage[1+(i-1)*time_steps:i*time_steps] = t.advantage
        traj.prob[1+(i-1)*time_steps:i*time_steps,:]    = t.prob
        traj.reward[1+(i-1)*time_steps:i*time_steps]    = t.reward
        traj.state[1+(i-1)*time_steps:i*time_steps,:]   = t.state
        traj.value[1+(i-1)*time_steps:i*time_steps]     = t.value
    end
    return traj
end

function create_mini_batches(
    sample_length::Int64, 
    batch_size::Int64
    )::Vector{Vector{Int64}}

    rng2 = StableRNG(rand(rng,1:1_000_000))
 
    ind = shuffle(rng2,collect(1:sample_length))
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
    optimizer_actor::Flux.Optimise.AbstractOptimiser,
    optimizer_critic::Flux.Optimise.AbstractOptimiser,
    global_traj::Trajectory,
    global_actor::Chain,
    global_critic::Chain,
    n_epochs::Int64,
    batch_size::Int64,
    c_1::Float64,
    c_2::Float64,
    β::Float64,
    ϵ::Float64,
    log::Logger
    )

    NT = length(global_traj.advantage)
    
    for k in 1:n_epochs
        batch_ind = create_mini_batches(NT, batch_size)
        
        for ind in batch_ind

            adv = global_traj.advantage[ind]
            adv = (adv.-mean(adv))./std(adv)

            target_value = global_traj.reward[ind]+adv

            states = transpose(global_traj.state[ind,:])
            actions = global_traj.action[ind]
            prob_old = global_traj.prob[ind,:]

            prob_old_a = [prob_old[i,a] for (i,a) in enumerate(actions)]

            ps_actor = Flux.params(global_actor)
            gs_actor = gradient(ps_actor) do
                prob = global_actor(states)

                prob = transpose(logsoftmax(prob))

                prob_a = [prob[i,a] for (i,a) in enumerate(actions)]
                
                ratio = exp.(prob_a .- prob_old_a)

                surr_1 = ratio.*adv
                surr_2 = clamp.(ratio,1-ϵ,1+ϵ).*adv

                L_clip = mean(min.(surr_1, surr_2))

                L_entropy = entropy(exp.(prob))
                kl = Flux.kldivergence(exp.(prob),exp.(prob_old))

                ChainRules.ignore_derivatives() do
                    log.L_entropy[log.index]     = c_2*L_entropy
                    log.L_clip[log.index]        = -L_clip
                    log.kl_divergence[log.index] = kl
                    log.L_total[log.index]       += -(L_clip + c_2*L_entropy - β*kl)
                end

                -(L_clip + c_2*L_entropy - β*kl)
            end

            clip_by_global_norm!(gs_actor, ps_actor, 0.5f0)
            Flux.update!(optimizer_actor, ps_actor, gs_actor)

            ps_critic = Flux.params(global_critic)
            gs_critic = gradient(ps_critic) do        

                value = global_critic(states)

                L_vf = 0.5*mean((value .- target_value).^2)

                ChainRules.ignore_derivatives() do
                    log.L_value[log.index] = c_1*L_vf
                    log.L_total[log.index] -= c_1*L_vf
                    log.index              += 1
                end

                c_1*L_vf
            end
            
            clip_by_global_norm!(gs_critic, ps_critic, 0.5f0)
            Flux.update!(optimizer_critic, ps_critic, gs_critic)
        end
    end
end

function update_worker(
    global_actor::Chain, 
    global_critic::Chain, 
    worker::Worker
    )::Worker

    worker.agent.actor = deepcopy(global_actor)
    worker.agent.critic = deepcopy(global_critic)
    
    return worker
end

function test_actor(
    actor::Chain,
    environment::AbstractEnv
    )
    rewards = []
    for _ in 1:20
        reset!(environment)
        tmp = []    
        while !is_terminated(environment)
            s = state(environment)
            
            probs = logsoftmax(actor(s))
            a = wsample(rng,collect(1:length(probs)),exp.(probs))
            
            environment(a)
            push!(tmp, reward(environment))
        end
        push!(rewards, sum(tmp))
    end
    return mean(rewards), minimum(rewards), maximum(rewards)
end

function main()

    iterations  = 250
    time_steps  = 2048
    n_envs      = 1
    n_epochs    = 10
    batch_size  = 64*n_envs
    c_1         = 0.5
    c_2         = 0.05
    β           = 0.0
    ϵ           = 0.2

    environment = CartPoleEnv(rng=rng)
    num_actions = length(action_space(environment))
    num_states  = length(state_space(environment))

    actor = Chain(Dense(num_states => 64,tanh; init=orthogonal(rng)),
                    Dense(64 => num_actions; init=orthogonal(rng)))

    critic = Chain(Dense(num_states => 64,tanh; init=orthogonal(rng)),
                    Dense(64 => 1; init=orthogonal(rng)))

    optimizer_actor  = Adam(3e-5)
    optimizer_critic = Adam(3e-5)

    workers = [Worker(time_steps,CartPoleEnv(rng = StableRNG(hash(1+i))),actor,critic) for i in 1:n_envs]

    logger = Logger(Int64(iterations*time_steps*n_envs*n_epochs/batch_size))
    
    r_baseline = []
    span_baseline = ([],[])
    for i in 1:iterations
        rew, min_r, max_r = test_actor(actor,environment)
        push!(r_baseline, rew)
        push!(span_baseline[1],min_r)
        push!(span_baseline[2],max_r)
    end

    r = []
    span = ([],[])
    iter = ProgressBar(1:iterations)

    for i in iter

        workers = workers .|> run_policy .|> calculate_advantage_estimate

        traj = collect_trajectories(workers)

        update_params(optimizer_actor,optimizer_critic,traj,actor,critic,n_epochs,batch_size,c_1,c_2,β,ϵ,logger)
        workers = [update_worker(actor,critic,w) for w in workers]
        rew, min_r, max_r = test_actor(actor, environment)
        
        push!(r, rew)
        push!(span[1],min_r)
        push!(span[2],max_r)
    end

    p1 = plot(range(1,iterations), r; label="Trained agent", title="Rewards", ribbon=span)
    plot!(range(1,iterations), r_baseline; label="Baseline", ribbon=span_baseline,legend=:topleft)
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