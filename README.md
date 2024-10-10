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

The clipped loss function represents the loss of the the actor, to get the full loss function used in the PPO algorithm we also need to represent the loss of the critic as well as an entropy term to encourage exploring. For the critic (this loss is also refered to as the value function loss) the original implementation uses root mean square error, in my implementation I chose to use the Huber loss instead as it is more robust. We now define the full loss which we want to minimize as

$$L^{CLIP+VF+S}(\theta)=\mathbb{E}_t\left[-L^{CLIP}(\theta)+c_1 L^{VF}(\theta)-c_2S\left[ \pi{\theta}\right]\right]$$

## Implementation details
### Bootstrapping advantage

I found that it is really important to bootstrap the advantage of the next step in the environment if the environment is not finished after the maximum number of time steps. This is done by letting your critic network predict what the next value will be. If this is not done then the agent cannot utilize advantages beyond the maximum reward for the maximum number of time steps.

### Utilizing vectorisation

This might seem as a no-brainer since this is one of the key strengths of algorithm but I foolishly did not vectorize the gathering of data at the beginning. My implementation of this is still not the prettiest but implementing this will save you a lot of time.

### Plot KL-divergence 

This is useful both as a debugging tool and hyperparameter tuning tool. At the first epoch in each iteration the KL-divergence should be zero since you have not updated your networks yet, if it is not you have a bug. If you have large KL-divergence then your policy is changing too much and you will most likely suffer from instability.

### Activation function

The original paper used the tanh activation function but I decided to use relu instead. I found that for the cartpole environment tanh lead to failed training. There is no guarantee that relu will work well in every environment but keep in mind that if the training returns poor results then maybe the activation function is the issue.

## Hyperparameters and results

I trained my agent on the cartpole- environment with 500 steps as maximum number of steps before terminating the environment. I used two separate networks for my critic and actor with two hidden layers in each. I have not explained all parameters in this text but I have adhered to the naming scheme of the original paper. If you are curious about for instance $\gamma$ I refer to the origninal paper.

- Adam optimizer learning rate: $2.5\times10^{-4}$
- Iterations: 1500
- Maximum time steps: 128
- Number of parallel environments: 4
- Number of epochs: 10
- Mini-batch size: 128
- $c_1$ : 0.5
- $c_2$ : 0.01
- $\epsilon$ : 0.2
- $\beta$ : 0.0 (coefficient for KL-divergence penalty in the loss, not used)
- $\gamma$ : 0.99
- $\lambda$ : 0.95
- Network hidden layer size: 64
### Metrics
<p align="center">
<img src="metrics.svg" width="500">
</p>
The image shows the resulting metrics from my training. The rewards-metric shows the mean and total span of the achieved rewards of the agent over 10 test runs in the environment. We can see that after 1500 iterations, the agent on average achieves a reward of ca. 490/500. From the entropy and the KL-divergence we can see that the training has started to or finished converging with quite small changes being made to the policy each epoch.
