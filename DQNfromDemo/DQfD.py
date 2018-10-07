import sys
from os import path

local = path.abspath(__file__)
root = path.dirname(path.dirname(local))
if root not in sys.path:
    sys.path.append(root)

from Common.prioritized_memory import Memory, WeightedMSE
import torch
import math
import random
from torch import optim
from collections import defaultdict as ddict
from functools import reduce


class AbstractDeepQL:
    def __init__(self, Net, actionFinder=None, eps=0.9, lr=5e-3, gamma=0.9, mbsize=20, C=100, N=500, lambda1=1.0,
                 lambda2=1.0, lambda3=1e-5, n_step=3):
        self.exp = []
        self.eps = eps
        self.net = Net()
        self.opt = optim.Adam(self.net.parameters(), lr=lr, weight_decay=lambda3)
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
        self.ed = 0.005  # bonus for demonstration
        self.ea = 0.001
        self.margin = 0.8
        self.n_step = n_step
        self.lambda1 = lambda1  # n-step return
        self.lambda2 = lambda2  # supervised loss
        self.lambda3 = lambda3  # L2
        self.replay.e = 0
        self.demoReplay = ddict(list)

    def calcQ(self, net, s, A):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    def findMaxA(self, state, num=1):
        raise NotImplementedError

    def sample(self):
        return self.replay.sample(self.mbsize)

    def store(self, data):
        self.replay.add(data)

    def storeDemoTransition(self, s, a, r, s_, done, demoEpisode):
        s = torch.Tensor(s)
        s_ = torch.Tensor(s_)
        episodeReplay = self.demoReplay[demoEpisode]  # replay of certain demo episode
        index = len(episodeReplay)
        data = (s, a, r, s_, done, (demoEpisode, index))
        episodeReplay.append(data)
        self.store(data)

    def storeTransition(self, s, a, r, s_, done):
        s = torch.Tensor(s)
        s_ = torch.Tensor(s_)
        self.store((s, a, r, s_, done, None))

    def JE(self, samples):
        loss = torch.tensor(0.0)
        count = 0  # number of demo
        for s, aE, *_, isdemo in samples:
            if isdemo is None:
                continue
            A = self.findMaxA(s, 2)
            if len(A) == 1:
                continue
            QE = self.calcQ(self.net, s, aE)
            A1, A2 = A
            maxA = A2 if (A1 == aE).all() else A1
            Q = self.calcQ(self.net, s, maxA)
            if (Q + self.margin) < QE:
                continue
            else:
                loss += (Q - QE)
                count += 1
        return loss / count if count != 0 else loss

    def Jn(self, samples):
        loss = torch.tensor(0.0)
        count = 0
        for s, a, r, s_, done, isdemo in samples:
            if isdemo is None:
                continue
            episode, idx = isdemo
            nidx = idx + self.n_step
            lepoch = len(self.demoReplay[episode])
            if nidx > lepoch:
                continue
            count += 1
            ns, na, nr, ns_, ndone, _ = zip(*self.demoReplay[episode][idx:nidx])
            ns, na, ns_, ndone = ns[-1], na[-1], ns_[-1], ndone[-1]
            discountedR = reduce(lambda x, y: (x[0] + self.gamma ** x[1] * y, x[1] + 1), nr, (0, 0))[0]
            maxA = self.findMaxA(ns_)
            target = discountedR if ndone else discountedR + self.gamma ** self.n_step * self.calcQ(self.net2, ns_,
                                                                                                    maxA)
            predict = self.calcQ(self.net, s, a)
            loss += (target - predict) ** 2
        return loss / count

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
            e = self.ea if samples[i][-1] is None else self.ed
            self.replay.update(idxs[i], error + e)

        Jtd = self.loss(Qpredict, Qtarget, IS)
        JE = self.JE(samples)
        Jn = self.Jn(samples)
        #print(Jtd, JE)
        J = Jtd + self.lambda1 * Jn + self.lambda2 * JE
        J.backward()
        self.opt.step()
        #Qpredict, Qtarget = self.calcTD(samples)
        #Jtd = self.loss(Qpredict, Qtarget, IS)
        #JE = self.JE(samples)
        #print(Jtd, JE)
        #print()

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
                return net(s, A)[0]
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

    def findMaxA(self, state, num=1):
        net = self.net
        net.eval()
        A = self.actionFinder(state)
        Q = self.calcQ(self.net, state, A)
        AQ = list(zip(A, Q))
        AQ.sort(key=lambda x: -x[1])  # sort by Q from max to min

        if num != 1:
            return [aq[0] for aq in AQ[:num]]
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
            return Q[[i for i in range(len(A))], A]

    def act(self, state):
        state = torch.Tensor(state)
        if self.noisy:
            self.net.sample()
            return self.findMaxA(state)
        maxA = self.findMaxA(state)
        r = random.random()
        a = maxA if self.eps > r else random.sample(self.A, 1)
        return a

    def findMaxA(self, state, num=1):
        net = self.net
        net.eval()
        Q = self.calcQ(self.net, state, self.A)
        net.train()
        return [self.A[Q.argmax()]]
