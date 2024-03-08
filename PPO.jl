using Flux, Plots, ReinforcementLearning, Statistics, Random, ProgressBars, Base.Threads, StableRNGs
using StatsBase: entropy, wsample
theme(:ggplot2)

rng = StableRNG(1)

mutable struct ADAM
    eta::Float64
    beta::Tuple{Float64,Float64}
    epsilon::Float64
    state::IdDict
  end
  
  ADAM(η = 0.001, β = (0.9, 0.999), ϵ = 1e-5) = ADAM(η, β, ϵ, IdDict())
  
  function apply!(o::ADAM, x, Δ)
    η, β, ϵ = o.eta, o.beta, o.epsilon
  
    mt, vt, βp = get!(o.state, x) do
        (zero(x), zero(x), Float64[β[1], β[2]])
    end :: Tuple{typeof(x),typeof(x),Vector{Float64}}
  
    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ^2
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
    βp .= βp .* β
  
    return Δ
  end

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

function advantage_estimate(
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
    workers::Vector{Worker}
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

function update_params(
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

            target_value = adv .+ global_traj.value[ind]

            states = transpose(stack(global_traj.state[ind],dims=1))
            actions = global_traj.action[ind]
            prob_old = global_traj.prob[ind]
            prob_old = [prob_old[i][a] for (i,a) in enumerate(actions)]

            ps = Flux.params(global_actor, global_critic)
            gs = gradient(ps) do

                value = global_critic(states)
                prob = global_actor(states)
                prob_a = [prob[:,i][a] for (i,a) in enumerate(actions)]
                
                L_vf = mean((value .- target_value).^2)
                ratio = prob_a ./ prob_old

                surr_1 = ratio .* adv
                surr_2 = clamp.(ratio,1-ϵ,1+ϵ).*adv

                L_clip = mean(min.(surr_1, surr_2))

                L_entropy = entropy(prob)

                L_clip + c_1*L_vf - c_2*L_entropy
            end

            clip_by_global_norm!(gs, ps, 0.5f0)
            Flux.update!(optimizer, ps, gs)
        end
    end
end

function reset_worker(global_actor::Chain, global_critic::Chain, worker::Worker)
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
            
            probs = actor(s)
            a = wsample(rng,collect(1:length(probs)),probs)
            
            environment(a)
            push!(tmp, reward(environment))
        end
        push!(rewards, sum(tmp))
    end
    return mean(rewards), minimum(rewards), maximum(rewards)
end

function main()

    iterations = 100
    time_steps = 128
    n_envs = 4

    n_epochs    = 10
    batch_size  = 64
    c_1         = 0.5
    c_2         = 0.01
    ϵ           = 0.2

    environment = CartPoleEnv(rng=rng)
    num_actions = length(action_space(environment))
    num_states =  length(state_space(environment))

    actor::Chain = Chain(Dense(num_states => 64,tanh; init=glorot_uniform(rng)), 
                    Dense(64 => 64,tanh; init=glorot_uniform(rng)),  
                    Dense(64 => num_actions; init=glorot_uniform(rng)), 
                    softmax)

    critic::Chain = Chain(Dense(num_states => 64,tanh; init=glorot_uniform(rng)), 
                    Dense(64 => 64,tanh; init=glorot_uniform(rng)), 
                    Dense(64 => 1; init=glorot_uniform(rng)))

    optimizer = Adam(2.5e-4)

    workers = [Worker(Trajectory(),Agent(actor,critic),CartPoleEnv(rng = StableRNG(hash(1+i))), time_steps) for i in 1:n_envs]
    r = []
    span = ([],[])
    iter = ProgressBar(1:iterations)

    for i in iter

        Threads.@threads for i in 1:n_envs
            workers[i] = workers[i] |> run_policy |> advantage_estimate
        end

        traj = collect_trajectories(workers)

        update_params(optimizer,traj,actor,critic,n_epochs,batch_size, c_1, c_2, ϵ)

        workers = [reset_worker(actor,critic,w) for w in workers]
        rew, s, b = test_actor(actor, environment)
        
        push!(r, rew)
        push!(span[1],s)
        push!(span[2],b)
    end  

    plot(range(1,iterations), r; ribbon=span)
end

main()