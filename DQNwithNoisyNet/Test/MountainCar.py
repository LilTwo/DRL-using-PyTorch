import sys
from os import path
local=path.abspath(__file__)
root=path.dirname(path.dirname(path.dirname(local)))
if root not in sys.path:
    sys.path.append(root)

import gym
import torch
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.nn.functional as F
from DQNwithNoisyNet.NoisyLayer import NoisyLinear
from DQNwithNoisyNet import DQN_NoisyNet

class NoisyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_s = NoisyLinear(2, 40)
        self.fc1_a = NoisyLinear(1, 40)
        self.fc2 = NoisyLinear(40, 1)

    def forward(self, s, a):
        x = self.fc1_s(s) + self.fc1_a(a)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def sample(self):
        for layer in self.children():
            if hasattr(layer, "sample"):
                layer.sample()


class NoisyNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = NoisyLinear(2, 40)
        self.fc2 = NoisyLinear(40, 3)

    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def sample(self):
        for layer in self.children():
            if hasattr(layer, "sample"):
                layer.sample()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_s = nn.Linear(2, 40)
        self.fc1_a = nn.Linear(1, 40)
        self.fc2 = nn.Linear(40, 1)

    def forward(self, s, a):
        x = self.fc1_s(s) + self.fc1_a(a)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 40)
        self.fc2 = nn.Linear(40, 3)

    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    s = env.reset()
    s = torch.Tensor(s)
    A=[[0],[1],[2]]
    dqn = DQN_NoisyNet.DeepQL(NoisyNet, lr=0.001, gamma=0.9, N=10000, C=500, actionFinder=lambda x:A)
    #dqn = DQN_NoisyNet.DeepQLv2(NoisyNet2, lr=0.001, gamma=0.9, N=10000, C=500, actionFinder=lambda x:A)
    process = []
    epoch = 80
    eps_start = 0.05
    eps_end = 0.95
    N = 1 - eps_start
    lam = -math.log((1 - eps_end) / N) / epoch
    total = 0
    dqn.replay.beta_increment_per_sampling = 0

    for i in range(epoch):
        dqn.eps = 1 - N * math.exp(-lam * i)
        dqn.replay.beta = dqn.replay.beta + 1.1/epoch if dqn.replay.beta < 1 else 1

        total = 0
        print(i, dqn.eps, dqn.replay.beta)
        while True:
            if total % 5000 == 0:
                print("trianing process total:", total)
            a = dqn.act(s)
            s_, r, done, _ = env.step(a[0])
            total += 1
            r = 10 if done else -1
            dqn.storeTransition(s, a, r, s_, done)
            dqn.update()
            s=s_

            if done:
                s = env.reset()
                print('finish total:', total)
                process.append(total)
                break

    plt.plot(process)
    plt.show()
    env.close()

    # torch.save(dqn.net.state_dict(),"./model.txt")
    # dqn.eps=1
    total = 0
    # dqn.net.load_state_dict(torch.load("./model.txt"))
    s = env.reset()
    s = torch.Tensor(s)
    while True:
        a = dqn.act(s)[0]
        s, r, done, _ = env.step(a)
        total += 1
        s = torch.Tensor(s)
        env.render()
        if done:
            s = env.reset()
            s = torch.Tensor(s)
            print(total)
            total = 0

    env.close()
