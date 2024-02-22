using Flux, Plots, ReinforcementLearningBase
using Flux.Optimise: update!

mutable struct Trajectory
    state
    action 
    reward 
    values 
    done 
end

function create_network(
    nr_layers::Int64, 
    layer_size::Vector{Int64},
    activations::Vector{Function},
    )::Chain

    network::Vector{Dense} = [Dense(layer_size[i] => layer_size[i+1], activations[i]) for i in 1:(nr_layers-1)]

    return Chain(network...)
end

function advantage_estimate(trajectory)

end

function main()

    env = CartPoleEnv()
    num_actions, num_states = length(action_space), length(state_space)

    actor = create_network(4,[num_states,64,64,num_actions],[tanh,tanh,identity])
    critic = create_network(4,[num_states,64,64,1],[tanh,tanh,identity])

    optimiser = Adam()
    
end

main()