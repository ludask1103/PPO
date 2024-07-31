using Flux, Plots, ReinforcementLearning, Statistics, Random, ProgressBars, Base.Threads, StableRNGs, ChainRules
using StatsBase: entropy, wsample
theme(:ggplot2)

rng = StableRNG(1)

mutable struct Trajectory
    state     ::Vector{Vector{Float64}}
    action    ::Vector{Int64}  
    prob      ::Vector{Vector{Float64}}
    reward    ::Vector{Float64}
    value     ::Vector{Float64}
    done      ::Vector{Bool}   
    advantage ::Vector{Float64}
    
    function Trajectory()
        new(
            Float64[],
            Int64[],
            Float64[],
            Float64[],
            Float64[],
            Bool[],
            Float64[]
        )
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
end

function run_policy(
    worker::Worker
    )::Worker

    for _ in 1:worker.time_steps
        if is_terminated(worker.environment)
            reset!(worker.environment)
        end
        s = state(worker.environment)
        probs = worker.agent.actor(s)
        probs = softmax(probs)

        a = wsample(rng,collect(1:length(probs)),probs)
        worker.environment(a)

        r = reward(worker.environment)
        v = worker.agent.critic(s)[1]

        push!(worker.trajectory.state, s)
        push!(worker.trajectory.prob, probs)
        push!(worker.trajectory.action, a)
        push!(worker.trajectory.reward, r)
        push!(worker.trajectory.value, v)
        push!(worker.trajectory.done, is_terminated(worker.environment))
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

        pushfirst!(worker.trajectory.advantage, A)

        A_prime = A
    end

    return worker
end

function collect_trajectories(
    workers::Vector{Worker},
    γ::Float64
    )::Trajectory

    traj = Trajectory()

    for w in workers
        t = w.trajectory
        append!(traj.action, t.action)
        append!(traj.advantage, t.advantage)
        append!(traj.prob, t.prob)
        append!(traj.reward, t.reward)
        append!(traj.state, t.state)
        append!(traj.value, t.value)
    end

    traj.reward = calculate_discounted_rewards(traj.reward,γ)

    return traj
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

    return batch_ind
end

function calculate_discounted_rewards(rewards, γ)
    T = length(rewards)
    discounted_rewards = zeros(T)
    running_add = 0.0
    for t in T:-1:1
        running_add = rewards[t] + γ * running_add
        discounted_rewards[t] = running_add
    end

    return discounted_rewards
end

L_e, L_value, L_cl, L, r = [], [], [], [], []
function _update_params(
    optimizer,
    global_traj::Trajectory,
    global_actor::Chain,
    global_critic::Chain,
    n_epochs::Int64,
    batch_size::Int64,
    c_1::Float64,
    c_2::Float64,
    ϵ::Float64
    )    

    NT = length(global_traj.advantage)
    
    for k in 1:n_epochs
        batch_ind = create_mini_batches(NT, batch_size)
        
        for ind in batch_ind

            adv = global_traj.advantage[ind]
            adv = (adv.-mean(adv))./std(adv)

            target_value = global_traj.reward[ind]

            states = transpose(stack(global_traj.state[ind],dims=1))
            actions = global_traj.action[ind]
            prob_old = global_traj.prob[ind]
            prob_old = [prob_old[i][a] for (i,a) in enumerate(actions)]

            ps = Flux.params([global_actor, global_critic])
            gs = gradient(ps) do

                value = global_critic(states)
                prob = global_actor(states)

                prob = logsoftmax(prob)

                prob_a = [prob[:,i][a] for (i,a) in enumerate(actions)]
                
                L_vf = mean((value .- target_value).^2)
                ratio = exp.(prob_a .- prob_old)

                surr_1 = ratio .* adv
                surr_2 = clamp.(ratio,1-ϵ,1+ϵ).*adv

                L_clip = mean(min.(surr_1, surr_2))

                L_entropy = -sum(exp.(prob) .* prob) * 1 // size(prob, 2)

                ChainRules.ignore_derivatives() do 
                    push!(L_e,c_2*L_entropy)
                    push!(L_value,c_1*L_vf)
                    push!(L_cl,L_clip)
                    push!(L,-L_clip + c_1*L_vf - c_2*L_entropy)
                end

                -(L_clip - c_1*L_vf + c_2*L_entropy)
            end

            clip_by_global_norm!(gs, ps, 0.5f0)
            Flux.update!(optimizer, ps, gs)            
        end
    end
end
function update_params(
    optimizer_actor,
    optimizer_critic,
    global_traj::Trajectory,
    global_actor::Chain,
    global_critic::Chain,
    n_epochs::Int64,
    batch_size::Int64,
    c_1::Float64,
    c_2::Float64,
    ϵ::Float64
    )    

    NT = length(global_traj.advantage)
    
    for k in 1:n_epochs
        batch_ind = create_mini_batches(NT, batch_size)
        
        for ind in batch_ind

            adv = global_traj.advantage[ind]
            adv = (adv.-mean(adv))./std(adv)

            target_value = global_traj.reward[ind]+adv

            states = transpose(stack(global_traj.state[ind],dims=1))
            actions = global_traj.action[ind]
            prob_old = global_traj.prob[ind]
            prob_old = [prob_old[i][a] for (i,a) in enumerate(actions)]

            ps_actor = Flux.params(global_actor)
            gs_actor = gradient(ps_actor) do
                prob = global_actor(states)

                prob = logsoftmax(prob)

                prob_a = [prob[:,i][a] for (i,a) in enumerate(actions)]
                
                ratio = exp.(prob_a .- prob_old)

                surr_1 = ratio .* adv
                surr_2 = clamp.(ratio,1-ϵ,1+ϵ).*adv

                L_clip = mean(min.(surr_1, surr_2))

                L_entropy = -sum(exp.(prob) .* prob) * 1 // size(prob, 2)

                ChainRules.ignore_derivatives() do
                    append!(r,log.(mean(ratio)))
                    push!(L_e,-c_2*L_entropy)
                    push!(L_cl,L_clip)
                end

                (L_clip - c_2*L_entropy)
            end

            clip_by_global_norm!(gs_actor, ps_actor, 0.5f0)
            Flux.update!(optimizer_actor, ps_actor, gs_actor)

            ps_critic = Flux.params(global_critic)
            gs_critic = gradient(ps_critic) do               

                value = global_critic(states)
                
                L_vf = mean((value .- target_value).^2)
                ChainRules.ignore_derivatives() do 
                    push!(L_value,c_1*L_vf)
                end

                c_1*L_vf 
            end

            clip_by_global_norm!(gs_critic, ps_critic, 0.5f0)
            Flux.update!(optimizer_critic, ps_critic, gs_critic)            
        end
    end
end

function update_worker(global_actor::Chain, global_critic::Chain, worker::Worker)
    worker.agent.actor = deepcopy(global_actor)
    worker.agent.critic = deepcopy(global_critic)
    worker.trajectory = Trajectory()

    return worker
end

function test_actor(
    actor::Chain,
    environment::AbstractEnv
    )
    rewards = []
    for _ in 1:10
        reset!(environment)
        tmp = []    
        while !is_terminated(environment)
            s = state(environment)
            
            probs = logsoftmax(actor(s))
            a = wsample(rng,collect(1:length(probs)),probs)
            
            environment(a)
            push!(tmp, reward(environment))
        end
        push!(rewards, sum(tmp))
    end
    return mean(rewards), minimum(rewards), maximum(rewards)
end

function main()

    iterations  = 50
    time_steps  = 2048
    n_envs      = 10

    n_epochs    = 10
    batch_size  = 64
    c_1         = 0.5
    c_2         = 0.05
    ϵ           = 0.1

    environment = CartPoleEnv(rng=rng)
    num_actions = length(action_space(environment))
    num_states  =  length(state_space(environment))

    actor::Chain = Chain(Dense(num_states => 32,tanh; init=orthogonal(rng)), 
                    Dense(32 => num_actions; init=orthogonal(rng))
                    )

    critic::Chain = Chain(Dense(num_states => 32,tanh; init=orthogonal(rng)), 
                    Dense(32 => 1; init=orthogonal(rng)))

    optimizer_actor = Adam(3e-4)
    optimizer_critic = Adam(3e-4)

    workers = [Worker(Trajectory(),Agent(actor,critic),CartPoleEnv(rng = StableRNG(hash(1+i))), time_steps) for i in 1:n_envs]
    
    r_baseline = []
    span_baseline = ([],[])
    for i in 1:iterations
        rew, s, b = test_actor(actor,environment)
        push!(r_baseline, rew)
        push!(span_baseline[1],s)
        push!(span_baseline[2],b)
    end

    r = []
    span = ([],[])
    iter = ProgressBar(1:iterations)

    for i in iter

        workers = workers .|> run_policy .|> calculate_advantage_estimate

        traj = collect_trajectories(workers,0.99)

        update_params(optimizer_actor,optimizer_critic,traj,actor,critic,n_epochs,batch_size,c_1,c_2,ϵ)

        workers = [update_worker(actor,critic,w) for w in workers]
        rew, s, b = test_actor(actor, environment)
        
        push!(r, rew)
        push!(span[1],s)
        push!(span[2],b)
    end

    #p1 = plot(range(1,iterations), r, label="Trained agent", title="Rewards"; ribbon=span)
    #plot!(range(1,iterations), r_baseline,label="Baseline"; ribbon=span_baseline)
    p1 = plot(range(1,iterations), r, label="Trained agent", title="Rewards")
    plot!(range(1,iterations), r_baseline,label="Baseline")
    p2 = plot(1:length(L_e), L_e, title="Entropy Loss", legend=false)
    p3 = plot(1:length(L_e), L_value, title="Value Loss", legend=false)
    p4 = plot(1:length(L_e), L_cl, title="CLIP Loss", legend=false)
    #p5 = plot(1:length(L_e), L, title="Loss", legend=false)
    p5 = plot(1:length(r), r, title="ratio", legend=false)
    plot(p1,p2,p3,p4,p5,layout=(5,1),size=(700,800))
    
end

main()