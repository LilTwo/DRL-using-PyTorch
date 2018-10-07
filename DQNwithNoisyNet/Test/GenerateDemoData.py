from os import path
import sys
parent_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from DQNwithNoisyNet.NoisyLayer import NoisyLinear
from DQNwithNoisyNet import DQN_NoisyNet
import json


class NoisyNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = NoisyLinear(4, 40)
        self.fc2 = NoisyLinear(40, 2)

    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def sample(self):
        for layer in self.children():
            if hasattr(layer, "sample"):
                layer.sample()


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    s = env.reset()
    A = [[0], [1]]
    dqn = DQN_NoisyNet.DeepQLv2(NoisyNet2, noisy=True, lr=0.002, gamma=1, actionFinder=lambda x: A)
    dqn.net.load_state_dict(torch.load("./CartPoleExpert.txt"))
    total = 0
    s = env.reset()
    epoch = 3
    step = 0
    demo = {}
    for e in range(epoch):
        data=[]
        while True:
            a = dqn.act(s)[0]
            s_, r, done, _ = env.step(a)
            r = -1 if done and total < 500 else 0.002
            data.append([list(s),[int(a)],r,list(s_),done])
            total += 1
            s=s_
            if done:
                demo[e]=data
                s = env.reset()
                print(total)
                total = 0
                break
    with open("CartPoleDemo.txt","w") as file:
        file.write(json.dumps(demo))

    env.close()
