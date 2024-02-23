using Flux, Plots, ReinforcementLearning
using Flux.Optimise: update!

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

mutable struct training_process
    ep_wise::Vector{Float64}
    policy_wise::Vector{Float64}

    function training_process()
        new(
            Float64[],
            Float64[]
        )
    end
end

function create_network(
    nr_layers::Int64, 
    layer_size::Vector{Int64},
    activations::Vector{Function},
    )::Chain

    network::Vector{Dense} = [Dense(layer_size[i] => layer_size[i+1], activations[i]) for i in 1:(nr_layers-1)]

    return Chain(network...)
end

function run_policy(
    env::AbstractEnv, 
    actor::Chain,
    critic::Chain, 
    T::Int64
    )::Trajectory
    traj = Trajectory()

    for _ in 1:T
        if is_terminated(env)
            reset!(env)
        end
        s = state(env)
        probs = actor(s)
        a = argmax(probs)
        env(a)
        r = reward(env)
        v = critic(s)[1]

        push!(traj.state, s)
        push!(traj.prob, probs)
        push!(traj.action, a)
        push!(traj.reward, r)
        push!(traj.value, v)
        push!(traj.done, is_terminated(env))
    end

    return traj
end

function advantage_estimate(
    traj::Trajectory,
    γ::Float64=0.99,
    λ::Float64=0.95
    )::Trajectory
    
    A_prime = 0
    push!(traj.advantage, A_prime)

    T = length(traj.action)
    for i in T-1:-1:1
        mask = 1.0 - traj.done[i] #need to ignore next value if it is part of a new session 
        A_prime *= mask  
        δ = traj.reward[i] + γ*traj.value[i+1]*mask - traj.value[i]

        A = δ + γ*λ*A_prime

        pushfirst!(traj.advantage, A)

        A_prime = A
    end

    return traj
end

function calculate_loss(traj::Trajectory,
    c_1::Int64=1
    )::Vector{Vector{Float64}}

    
end

function update_params()

end

function main()
    env = CartPoleEnv()
    num_actions, num_states = length(action_space(env)), length(state_space(env))

    actor = create_network(4,[num_states,64,64,num_actions],[tanh,tanh,identity])
    critic = create_network(4,[num_states,64,64,1],[tanh,tanh,identity])

    optimiser = Adam()

    traj = run_policy(env,actor,critic,5) |> advantage_estimate
    println(traj)
    println(traj.advantage)
    println(traj.action)

end

main()