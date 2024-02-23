using Flux, Plots, ReinforcementLearningBase
using Flux.Optimise: update!

mutable struct Trajectory
    state   ::Vector{Float64}
    action  ::Vector{Int64}  
    prob    ::Vector{Float64}
    reward  ::Vector{Float64}
    values  ::Vector{Float64}
    done    ::Vector{Bool}   
    
    function Trajectory()
        new(
            Float64[],
            Int64[],
            Float64[],
            Float64[],
            Float64[],
            Bool[]
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
    T::Int64
    )::Trajectory
    traj = Trajectory()

    for i in 1:T
        
    end
end

function advantage_estimate(trajectory::Trajectory)::Vector{Float32}

end

function main()

    num_actions, num_states = length(action_space), length(state_space)

    env = CartPoleEnv()

    actor = create_network(4,[num_states,64,64,num_actions],[tanh,tanh,identity])
    critic = create_network(4,[num_states,64,64,1],[tanh,tanh,identity])

    optimiser = Adam()

    traj = Trajectory()
    
end

main()