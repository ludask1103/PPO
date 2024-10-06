# Proximal Policy Optimization (PPO)
Proximal policy optimization is a state-of-art reinforcement learning algorithm used in a wide variety of applications. This repository shows my attempt to implement it as well as I can in order to gain more knowledge into reinforcement learning and Julia. This text will present first some theory about PPO which will be followed by some implementation details and lessons I have learned throghout the project which I hope will help others in their attempts to implement the algorithm. <br>

# Quick Theory
PPO is an actor-critic on-policy algorithm which tries to maximize the improvement of the policy based on the data we have withouth taking a too large step and causing catastrophic interference. PPO achieves this by introducing a clipped loss function. Before diving into workings of the clipped loss we need to define some quantities. Let $\pi_{\theta}(a|s)$ define the policy for selecting action $a$ at state $s$ and $\pi_{\theta_{old}}(a|s)$ the previous iteration's policy. We then define the ratio as 

$$r_{t}(\theta) = {\pi_{\theta}(a_{t}|s_{t}) \over \pi_{\theta_{old}}(a_{t}|s_{t})}$$

If we then define the advantage at time step $t$ as $\hat{A}_{t}$ we can define the clipped loss function as 

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \ \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

$\epsilon$ is a clipping constant often set in the range [0.1,0.3]. By clipping the ratio in the loss function we prevent big differences between the new and old policy causing large changes in the policy and therefore unstable learning. <br>

The clipped loss function represents the loss of the the actor, to get the full loss function used in the PPO algorithm we also need to represent the loss of the critic as well as an entropy term to encourage exploring. For the critic (this loss is also refered to as the value function loss) the original implementation uses root mean square error, in my implementation I chose to use the Huber loss instead as it is more robust. We now define the full loss as

$$
L^{CLIP+VF+S}(\theta) = \mathbb{E}_t \left[ -L^{CLIP} + c_1 L^{VF} - c_2 S \left[ \pi_{\theta} \right] \right]
$$
