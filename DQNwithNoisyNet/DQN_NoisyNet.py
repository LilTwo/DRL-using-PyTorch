import random
from torch import optim
import torch
import math
import sys
from os import path
parent_dir = path.dirname(path.dirname(path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from Common.prioritized_memory import Memory, WeightedMSE
from Common.ValueCaculator import ValueCalculator1 as VC1
from Common.ValueCaculator import ValueCalculator2 as VC2


class DeepQL:
    def __init__(self, Net, actionFinder=None,eps=0.9, lr=5e-3, gamma=0.9, mbsize=20, C=100, N=500, L2=0):
        self.eps = eps
        self.gamma = gamma
        self.mbsize = mbsize
        self.C = C  # for target replacement
        self.c = 0
        self.replay = Memory(capacity=N)
        self.loss = WeightedMSE()
        self.actionFinder = actionFinder
        self.vc =VC1(Net,actionFinder) if actionFinder else VC2(Net)
        self.opt = optim.Adam(self.vc.predictNet.parameters(), lr=lr, weight_decay=L2)
        self.noisy = hasattr(self.vc.predictNet, "sample")
        # (state:tensor => Action :List[List])

    def act(self, state):
        state = torch.Tensor(state)
        A = self.vc.sortedA(state)
        if self.noisy:
            self.vc.predictNet.sample()
            return A[0]
        r = random.random()
        a = A[0] if self.eps > r else random.sample(A, 1)[0]
        return a

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
            self.vc.predictNet.sample()  # for choosing action
        alls, alla, allr, alls_, alldone, *_ = zip(*samples)
        maxA = [self.vc.sortedA(s_)[0] for s_ in alls_]
        if self.noisy:
            self.vc.predictNet.sample()  # for prediction
            self.vc.targetNet.sample()  # for target

        Qtarget = torch.Tensor(allr)
        Qtarget[torch.tensor(alldone) != 1] += self.gamma * self.vc.calcQ(self.vc.targetNet, alls_, maxA)[
            torch.tensor(alldone) != 1]
        Qpredict = self.vc.calcQ(self.vc.predictNet, alls, alla)
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
            self.vc.updateTargetNet()
        else:
            self.c += 1