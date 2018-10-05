import numpy as np
import random
from torch import optim
import torch
import math

if __package__:
    from .prioritized_memory import Memory, WeightedMSE
else:
    from prioritized_memory import Memory, WeightedMSE


class AbstractDeepQL:
    def __init__(self, Net, actionFinder=None,eps=0.9, lr=5e-3, gamma=0.9, mbsize=20, C=100, N=500, L2=0):
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
        self.noisy = hasattr(self.net, "sample")
        self.actionFinder = actionFinder
        # (state:tensor => Action :List[List])

    def calcQ(self, net, s, A):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    def findMaxA(self, state):
        raise NotImplementedError

    def sample(self):
        return self.replay.sample(self.mbsize)

    def store(self, data):
        self.replay.add(data)

    def storeTransition(self, s, a, r, s_, done):
        s = torch.Tensor(s)
        s_ = torch.Tensor(s_)
        self.store((s, a, r, s_, done))

    def calcTD(self, samples):
        if self.noisy:
            self.net.sample()  # for choosing action
        alls, alla, allr, alls_, alldone, *_ = zip(*samples)
        maxA = [self.findMaxA(s_) for s_ in alls_]
        if self.noisy:
            self.net.sample()  # for prediction
            self.net2.sample()  # for target

        Qtarget = torch.Tensor(allr)
        Qtarget[torch.tensor(alldone) != 1] += self.gamma * self.calcQ(self.net2, alls_, maxA)[
            torch.tensor(alldone) != 1]
        Qpredict = self.calcQ(self.net, alls, alla)
        return Qpredict, Qtarget

    def update(self):
        self.opt.zero_grad()
        samples, idxs, IS = self.sample()
        Qpredict, Qtarget = self.calcTD(samples)

        for i in range(self.mbsize):
            error = math.fabs(float(Qpredict[i] - Qtarget[i]))
            self.replay.update(idxs[i], error)

        J = self.loss(Qpredict, Qtarget, IS)
        J.backward()
        self.opt.step()

        if self.c >= self.C:
            self.c = 0
            self.net2.load_state_dict(self.net.state_dict())
            self.net2.eval()
        else:
            self.c += 1


class DeepQL(AbstractDeepQL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # (state:tensor => Action :List[List])

    def calcQ(self, net, s, A):
        # 1.single state, one or multiple actions
        # 2.muliplte states, one action per state
        if isinstance(s, torch.Tensor) and s.dim() == 1:  # situation 1
            A = torch.Tensor(A)
            if A.dim() == 1:
                return net(s,A)[0]
            return torch.Tensor([net(s, a) for a in A])

        if not isinstance(s, torch.Tensor):  # situation 2
            s = torch.stack(s)
            a = torch.Tensor(A)
            return net(s, a).squeeze()

    def act(self, state):
        state = torch.Tensor(state)
        A = self.actionFinder(state)
        if self.noisy:
            self.net.sample()
            return self.findMaxA(state)
        maxA = self.findMaxA(state)
        r = random.random()
        a = maxA if self.eps > r else random.sample(A, 1)[0]
        return a

    def findMaxA(self, state):
        net = self.net
        net.eval()
        A = self.actionFinder(state)
        Q = self.calcQ(self.net, state, A)
        net.train()
        return A[Q.argmax()]


# s => Q[s,a1], Q[s,a2]...
class DeepQLv2(AbstractDeepQL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        *_, last = self.net.children()
        self.A = list(range(last.out_features))

    def calcQ(self, net, s, A):
        # 1.single state
        # 2.muliplte states, one action per state
        if isinstance(s, torch.Tensor) and s.dim() == 1:  # situation 1
            return torch.Tensor([net(s)[a] for a in A])

        if not isinstance(s, torch.Tensor):  # situation 2
            s = torch.stack(s)
            Q = net(s)
            A = [a[0] for a in A]
            return Q[[i for i in range(len(A))],A]

    def act(self, state):
        state = torch.Tensor(state)
        if self.noisy:
            self.net.sample()
            return self.findMaxA(state)
        maxA = self.findMaxA(state)
        r = random.random()
        a = maxA if self.eps > r else random.sample(self.A, 1)
        return a

    def findMaxA(self, state):
        net = self.net
        net.eval()
        Q = self.calcQ(self.net, state, self.A)
        net.train()
        return [self.A[Q.argmax()]]
