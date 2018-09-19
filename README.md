# DRL-using-PyTorch
PyTorch implementations of Deep Reinforcement Algorithms

All the following DQN variation contain DDQN, prioritized replay, fixed target

## DQN with NoisyNet:  
https://github.com/LilTwo/DRL-using-PyTorch/tree/master/DQN_NoisyNet.  
reference：https://arxiv.org/pdf/1706.10295.pdf

NoisyNets add randomness to parameters of the network.  
With the presence of noisy layers, network has the ability to learn a domain-specific exploration strategy,  
rather than using epsilon-greedy and increase epsilon manualy during learning.  
From my expeience, NoisyNets usually need a smaller leaning rate than nomal nets to work well,  
and is way more sensitive to parameters's initial value.  
In MountainCar environment, there are some chance that the car never hit the top in first epsiode. 
I'm not sure whether this is because I wrote somthing wrong.

In the original paper, auothrs suggest that the summation of "sigma" can be view as the stochasticity of the layer.  
This have been implemented in "randomness" method of "NoisyLinear" class, but I only account "sigma" of bias term,  
because the weight matrix term involves input from the last layer which can change dramaticaly over epoch.  
And before the summation each "sigma" is normalized with "mu", because the magnitude of "sigma" is only meaningful when compares to "mu".  
