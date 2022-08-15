# 1 DRL-using-PyTorch
PyTorch implementations of Deep Reinforcement Algorithms

All the following DQN variations are derived from DQNwithNoisyNet folder and already contain DDQN, prioritized replay, fixed target　network.

## DQN with NoisyNet:  
https://github.com/LilTwo/DRL-using-PyTorch/tree/master/DQN_NoisyNet.  
reference:https://arxiv.org/pdf/1706.10295.pdf

NoisyNets add randomness to the parameters of the network.  
With the presence of noisy layers, network is able to learn a domain-specific exploration strategy,  
rather than using epsilon-greedy and increase epsilon manualy during training.  
From my expeience, a NoisyNet usually needs a smaller leaning rate than nomal nets to work well,  
and is very sensitive to parameters's initial value.  
In MountainCar environment, there are some chances that the car never hit the top in first epsiode. 
I'm not sure whether this is because I wrote somthing wrong.

In the original paper, auothrs suggest that the summation of "sigma" can be viewed as the stochasticity of the layer.  
This have been implemented in "randomness" method of "NoisyLinear" class with one modification: each "sigma" is normalized by "mu" before the summation.  

## DQN from Demonstrations (DQfD)
https://github.com/LilTwo/DRL-using-PyTorch/tree/master/DQNfromDemo  
reference:https://arxiv.org/pdf/1704.03732.pdf

If there are some expert's demonstrations produced by human or another well-trained agent, one may expect these data could speed up the training process by saving time from random exploration in a large state/action space.   
DQfD proivds a method to leverage demonstration data by pre-training the model on the demonstartion data solely before it starts to interact with the environment.  

## Hindsight Experience Replay (HER)
code will be uploaded soon.  
reference:https://papers.nips.cc/paper/7090-hindsight-experience-replay.pdf

Since model-free RL algorithms like DQN know nothing about the environment, they usually need lots of exploration to find out what is good or bad at the begining, especially when dealing with sparse reward.  
At the first few epochs of training, an agent is likely to get no positive reward during the whole episode, HER can make good use of these trajectorys by storing each trajectory in the replay buffer again but with different goals which are achieved by some states in the trajectory.    
So you can know for sure that there have some transition with positive reward are stored in the replay buffer after every episode is finished.  
The key for HER to work is that these goals should be correlated in a resonable way so that learning to behave well on one of them can also help to behave well on another one, so I express reservations about the authors opinion: using HER requires less domain knowledge than redefining a shaped reward.
