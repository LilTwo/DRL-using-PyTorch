# DRL-using-PyTorch
PyTorch implementations of Deep Reinforcement Algorithms

All the following DQN variations are derived from DQNwithNoisyNet folder and already contain DDQN, prioritized replay, fixed target　network.

## DQN with NoisyNet:  
https://github.com/LilTwo/DRL-using-PyTorch/tree/master/DQN_NoisyNet.  
reference：https://arxiv.org/pdf/1706.10295.pdf

NoisyNets add randomness to the parameters of the network.  
With the presence of noisy layers, network is able to learn a domain-specific exploration strategy,  
rather than using epsilon-greedy and increase epsilon manualy during training.  
From my expeience, a NoisyNet usually needs a smaller leaning rate than nomal nets to work well,  
and is very sensitive to parameters's initial value.  
In MountainCar environment, there are some chances that the car never hit the top in first epsiode. 
I'm not sure whether this is because I wrote somthing wrong.

In the original paper, auothrs suggest that the summation of "sigma" can be view as the stochasticity of the layer.  
This have been implemented in "randomness" method of "NoisyLinear" class with one difference: each "sigma" is normalized by "mu" before the summation.  

## DQN from Demonstrations (DQfD)
https://github.com/LilTwo/DRL-using-PyTorch/tree/master/DQNfromDemo  
reference：https://arxiv.org/pdf/1704.03732.pdf

Since model-free RL algorithms like DQN know nothing about the environment, they usually need lots of exploration to find out what is good or bad at the begining, especially when dealing with large state/action space and sparse reward.  
So it would be nice if there are some demonstrations produced by human or another well-trained agent.
DQfD proivded a method to leverage these demonstrations by first pre-training the model on the demonstartion data only before it starts to interact with the environment.  
