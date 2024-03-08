using ReinforcementLearning, Flux, Flux.Losses

UPDATE_FREQ = 64
n_env = 8
env = MultiThreadEnv([CartPoleEnv() for i in 1:n_env])
num_states, num_actions = length(state(env[1])), length(action_space(env[1]))
agent = Agent(
    policy = PPOPolicy(
        approximator = ActorCritic(
            actor = Chain(
                Dense(num_states, 64, tanh),
                Dense(64, 64, tanh),
                Dense(64, num_actions)
            )
            ,
            critic = Chain(
                Dense(num_states, 64, tanh),
                Dense(64, 64, tanh),
                Dense(64, 1)
            ),
            optimizer = Adam(3e-4)
        ) |> cpu,
        γ = 0.99f0,
        λ = 0.95f0,
        clip_range = 0.2f0,
        n_microbatches = 4,
        n_epochs = 10,
        actor_loss_weight = 1.0f0,
        critic_loss_weight = 1.0f0,
        entropy_loss_weight = 0.0f0,
        #dist = Categorical, This needs to be changed to Normal for continous action spaces and the actor and critic to GaussianNetworks
        update_freq = UPDATE_FREQ
    ),
    trajectory = PPOTrajectory(; 
    capacity = UPDATE_FREQ,
    state = Matrix{Float32} => (num_states, n_env),
    action = Vector{Int} => (n_env,),
    action_log_prob = Vector{Float32} => (n_env,),
    reward = Vector{Float32} => (n_env,),
    terminal = Vector{Bool} => (n_env,),
    )
)
ReinforcementLearningCore._run(agent, env, StopAfterStep(100),TotalBatchRewardPerEpisode())