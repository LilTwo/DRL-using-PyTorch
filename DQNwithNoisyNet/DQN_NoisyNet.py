import numpy as np
import random
from torch import optim
import torch
import math

if __package__:
    from .prioritized_memory import Memory, WeightedMSE
else:
    from prioritized_memory import Memory, WeightedMSE


# (s,a) => Q(s,a)
class DeepQL:
    def __init__(self, Net, noisy=True, eps=0.9, lr=5e-3, gamma=0.9, mbsize=20, C=100, N=500, L2=0, actionFinder=None):
        self.exp = []
        self.eps = eps
        self.net = Net()
        self.opt = optim.Adam(self.net.parameters(), lr=lr, weight_decay=L2)
        self.gamma = gamma
        self.mbsize = mbsize
        self.net2 = Net()
        self.net2.load_state_dict(self.net.state_dict())
        self.net2.eval()
        self.C = C  # for target replacement
        self.c = 0
        self.replay = Memory(capacity=N)
        self.loss = WeightedMSE()
        self.actionFinder = actionFinder
        self.noisy = noisy
        # (state:tensor => Action :List[List])

    def act(self, state):
        state = torch.Tensor(state)
        A = self.actionFinder(state)
        if self.noisy:
            self.net.sample()
            maxA = self.findMaxA(state)
            return maxA
        maxA = self.findMaxA(state)
        r = random.random()
        a = maxA if self.eps > r else random.sample(A, 1)[0]

        return a

    def sample(self):
        return self.replay.sample(self.mbsize)

    def store(self, data, error):
        self.replay.add(error, data)

    def findMaxA(self, state):
        net = self.net
        A = self.actionFinder(state)

        net.eval()
        Q = [net(state, a) for a in torch.Tensor(A)]
        Q = torch.Tensor(Q)
        net.train()
        return A[Q.argmax()]

    def storeTransition(self, s, a, r, s_, done):
        s = torch.Tensor(s)
        s_ = torch.Tensor(s_)
        error = self.calcError((s, a, r, s_, done))
        self.store((s, a, r, s_, done), error)

    def calcError(self, sample):
        s, a, r, s_, done = sample
        a = torch.Tensor(a)
        if self.noisy:
            self.net.sample()
        maxA = torch.Tensor(self.findMaxA(s_))
        if self.noisy:
            self.net.sample()
            self.net2.sample()
        target = r if done else r + self.gamma * self.net2(s_, maxA)
        error = self.net(s, a) - target
        error = float(error)
        return math.fabs(error)

    def update(self):
        self.opt.zero_grad()

        samples, idxs, IS = self.sample()
        if self.noisy:
            self.net.sample()  # for choosing action
        maxA = [self.findMaxA(s[3]) for s in samples]
        maxA = torch.Tensor(maxA)
        s, a, *_ = zip(*samples)
        s = torch.stack(s)
        a = torch.Tensor(a)
        if self.noisy:
            self.net.sample()  # for prediction
            self.net2.sample()  # for estimating Q
        predict = self.net(s, a)[:, 0]
        look_ahead = [r if done else r + self.gamma * self.net2(s_, maxA[i]) for i, (s, a, r, s_, done) in
                      enumerate(samples)]
        target = torch.Tensor(look_ahead)

        errors, ls = self.loss(predict, target, IS)
        ls.backward()
        for i in range(self.mbsize):
            self.replay.update(idxs[i], errors[i])

        self.opt.step()

        if self.c >= self.C:
            self.c = 0
            self.net2.load_state_dict(self.net.state_dict())
            self.net2.eval()
        else:
            self.c += 1


# s => Q[s,a1], Q[s,a2]...
class DeepQLv2:
    def __init__(self, Net, noisy=True, eps=0.9, lr=5e-3, gamma=0.9, mbsize=20, C=100, N=500, L2=0, actionFinder=None):
        self.exp = []
        self.net = Net()
        self.opt = optim.Adam(self.net.parameters(), lr=lr, weight_decay=L2)
        self.gamma = gamma
        self.mbsize = mbsize
        self.net2 = Net()
        self.net2.load_state_dict(self.net.state_dict())
        self.net2.eval()
        self.C = C
        self.c = 0
        self.replay = Memory(capacity=N)
        self.loss = WeightedMSE()
        self.eps = eps
        self.noisy = noisy
        self.actionFinder = actionFinder
        *_,last=self.net.children()
        self.A = list(range(last.out_features))

    def act(self, state):
        # state:list[float] A:list[list]
        state = torch.Tensor(state)
        if self.noisy:
            self.net.sample()
            a = self.findMaxA(state)
            return list(np.array(a))
        maxA = self.findMaxA(state)
        r = random.random()
        a = maxA if self.eps > r else random.sample(self.A, 1)
        return a

    def sample(self):
        return self.replay.sample(self.mbsize)

    def store(self, data, error):
        # data (s:tensor,a:list,r:scalar,s_:tensor,done:bool)
        self.replay.add(error, data)

    def findMaxA(self, state):
        net = self.net
        net.eval()
        Q = net(state)
        net.train()
        return [int(Q.argmax())]

    def storeTransition(self, s, a, r, s_, done):
        s = torch.Tensor(s)
        s_ = torch.Tensor(s_)
        error = self.calcError((s, a, r, s_, done))
        self.store((s, a, r, s_, done), error)

    def calcError(self, sample):
        s, a, r, s_, done = sample
        if self.noisy:
            self.net.sample()
        maxA = self.findMaxA(s_)
        if self.noisy:
            self.net.sample()
            self.net2.sample()
        target = r if done else r + self.gamma * self.net2(s)[maxA[0]]
        error = self.net(s)[a[0]] - target
        error = float(error)
        return math.fabs(error)

    def update(self):
        self.opt.zero_grad()

        samples, idxs, IS = self.sample()
        if self.noisy:
            self.net.sample()  # for choosing action
        maxA = [self.findMaxA(s[3]) for s in samples]
        s, a, *_ = zip(*samples)
        s = torch.stack(s)
        if self.noisy:
            self.net.sample()  # for prediction
            self.net2.sample()  # for estimating Q
        predict = [self.net(s[i])[a[i][0]] for i in range(self.mbsize)]
        look_ahead = [r if done else r + self.gamma * self.net2(s_)[maxA[i][0]] for i, (s, a, r, s_, done) in
                      enumerate(samples)]
        target = torch.Tensor(look_ahead)

        errors, ls = self.loss(predict, target, IS)
        ls.backward()
        for i in range(self.mbsize):
            self.replay.update(idxs[i], errors[i])

        self.opt.step()

        if self.c >= self.C:
            self.c = 0
            self.net2.load_state_dict(self.net.state_dict())
            self.net2.eval()
        else:
            self.c += 1
