import sys
from os import path

local = path.abspath(__file__)
root = path.dirname(path.dirname(local))
if root not in sys.path:
    sys.path.append(root)

from Common.prioritized_memory import Memory, WeightedMSE
from Common.ValueCaculator import ValueCalculator1 as VC1
from Common.ValueCaculator import ValueCalculator2 as VC2
import torch
import math
import random
from torch import optim
from collections import defaultdict as ddict
from functools import reduce
import numpy as np


class DeepQL:
    def __init__(self, Net, actionFinder=None, eps=0.9, lr=5e-3, gamma=0.9, mbsize=20, C=100, N=500, lambda1=1.0,
                 lambda2=1.0, lambda3=1e-5, n_step=3):
        self.eps = eps  # eps-greedy
        self.gamma = gamma  # discount factor
        self.mbsize = mbsize  # minibatch size
        self.C = C  # frequenct of target replacement
        self.c = 0  # target replacement counter
        self.replay = Memory(capacity=N)
        self.loss = WeightedMSE()
        self.actionFinder = actionFinder  # (state:tensor => Action :List[List])
        self.vc = VC1(Net, actionFinder) if actionFinder else VC2(Net)
        self.opt = optim.Adam(self.vc.predictNet.parameters(), lr=lr, weight_decay=lambda3)
        self.noisy = hasattr(self.vc.predictNet, "sample")
        self.ed = 0.005  # bonus for demonstration
        self.ea = 0.001
        self.margin = 0.8
        self.n_step = n_step
        self.lambda1 = lambda1  # n-step return
        self.lambda2 = lambda2  # supervised loss
        self.lambda3 = lambda3  # L2
        self.replay.e = 0
        self.demoReplay = ddict(list)

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

    def JE(self, samples):
        loss = torch.tensor(0.0)
        count = 0  # number of demo
        for s, aE, *_, isdemo in samples:
            if isdemo is None:
                continue
            A = self.vc.sortedA(s)
            if len(A) == 1:
                continue
            QE = self.vc.calcQ(self.vc.predictNet, s, aE)
            A1, A2 = np.array(A)[:2]  # action with largest and second largest Q
            maxA = A2 if (A1 == aE).all() else A1
            Q = self.vc.calcQ(self.vc.predictNet, s, maxA)
            if (Q + self.margin) < QE:
                continue
            else:
                loss += (Q - QE)
                count += 1
        return loss / count if count != 0 else loss

    def Jn(self, samples, Qpredict):
        # wait for refactoring, can't use with noisy layer
        loss = torch.tensor(0.0)
        count = 0
        for i,(s, a, r, s_, done, isdemo) in enumerate(samples):
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
            maxA = self.vc.sortedA(ns_)[0]
            target = discountedR if ndone else discountedR + self.gamma ** self.n_step * self.vc.calcQ(
                self.vc.targetNet, ns_,
                maxA)
            predict = Qpredict[i]
            loss += (target - predict) ** 2
        return loss / count

    def update(self):
        self.opt.zero_grad()
        samples, idxs, IS = self.sample()
        Qpredict, Qtarget = self.calcTD(samples)

        for i in range(self.mbsize):
            error = math.fabs(float(Qpredict[i] - Qtarget[i]))
            self.replay.update(idxs[i], error)

        Jtd = self.loss(Qpredict, Qtarget, IS*0+1)
        JE = self.JE(samples)
        Jn = self.Jn(samples,Qpredict)
        J = Jtd + self.lambda2 * JE + self.lambda1 * Jn
        J.backward()
        self.opt.step()

        if self.c >= self.C:
            self.c = 0
            self.vc.updateTargetNet()
        else:
            self.c += 1
