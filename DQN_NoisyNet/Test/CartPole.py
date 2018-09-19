import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.nn.functional as F
from NoisyLayer import NoisyLinear
import DQN_NoisyNet
from operator import methodcaller


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_s = nn.Linear(4, 40)
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
        self.fc1 = nn.Linear(4, 40)
        self.fc2 = nn.Linear(40, 2)

    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class NoisyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_s = NoisyLinear(4, 40)
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

    process = []
    randomness = []
    epoch = 200
    eps_start = 0.05
    eps_end = 0.95
    N = 1 - eps_start
    lam = -math.log((1 - eps_end) / N) / epoch
    total = 0
    count = 0  # successful count

    for i in range(epoch):
        print(i)
        if dqn.noisy:
            m = methodcaller("randomness")
            *_, rlast = map(m, dqn.net.children())  # only record the randomness of last layer
            randomness.append(rlast)
        dqn.eps = 1 - N * math.exp(-lam * i)
        count = count + 1 if total >= 500 else 0
        if count >= 2:
            dqn.eps = 1
            break
        total = 0
        while True:
            a = dqn.act(s)
            s_, r, done, _ = env.step(a[0])
            total += r
            r = -1 if done and total < 500 else 0.002
            dqn.storeTransition(s, a, r, s_, done)
            dqn.update()
            s = s_
            if done:
                s = env.reset()
                print('total:', total)
                process.append(total)
                break

    if dqn.noisy:
        plt.plot(randomness)
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
