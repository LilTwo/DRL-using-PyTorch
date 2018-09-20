import sys
from os import path

local = path.abspath(__file__)
root = path.dirname(path.dirname(local))
if root not in sys.path:
    sys.path.append(root)

from DQNwithNoisyNet import DQN_NoisyNet
import torch


class DeepQL(DQN_NoisyNet.DeepQL):
    def __init__(self, *args,lambda1=1.0,lambda2=1.0,lambda3=1e-5, **kwargs,):
        super().__init__(*args, **kwargs,L2=lambda3)
        self.ed = 1.0  # bonus for demonstration
        self.ea = 0.001
        self.margin = 0.8
        self.lambda1 = lambda1  # n-step return
        self.lambda2 = lambda2  # supervised loss
        self.lambda3 = lambda3  # L2
        self.replay.e = 0

    def storeTransition(self, s, a, r, s_, done, isdemo):
        s = torch.Tensor(s)
        s_ = torch.Tensor(s_)
        error = self.calcError((s, a, r, s_, done))
        e = self.ed if isdemo else self.ea
        self.store((s, a, r, s_, done, isdemo), error + e)

    def JE(self, samples):
        loss = torch.tensor(0.0)
        for s, a, *_, isdemo in samples:
            if not isdemo:
                continue
            QE = self.net(s, torch.Tensor(a))[0]
            Q = self.net(s, torch.Tensor(self.findMaxA(s)))[0]
            Q = QE if Q + self.margin < QE else Q
            loss += self.lambda2 * (Q - QE)
        return loss / self.mbsize

    def update(self):
        self.opt.zero_grad()
        samples, idxs, IS = self.sample()
        if self.noisy:
            self.net.sample()  # for choosing action
        maxA = [self.findMaxA(s[3]) for s in samples]
        maxA = torch.Tensor(maxA)
        s, a, *_, isdemo = zip(*samples)
        s = torch.stack(s)
        a = torch.Tensor(a)
        if self.noisy:
            self.net.sample()  # for prediction
            self.net2.sample()  # for estimating Q
        predict = self.net(s, a)[:, 0]
        look_ahead = [r if done else r + self.gamma * self.net2(s_, maxA[i]) for i, (s, a, r, s_, done, isdemo) in
                      enumerate(samples)]
        target = torch.Tensor(look_ahead)

        errors, ls = self.loss(predict, target, IS)
        if self.noisy:
            self.net.sample()
        ls += self.JE(samples)
        ls.backward()
        for i in range(self.mbsize):
            e = self.ed if isdemo[i] else self.ea
            self.replay.update(idxs[i], errors[i] + e)

        self.opt.step()
        if self.c >= self.C:
            self.c = 0
            self.net2.load_state_dict(self.net.state_dict())
            self.net2.eval()
        else:
            self.c += 1


class DeepQLv2(DQN_NoisyNet.DeepQLv2):
    def __init__(self, *args,lambda1=1.0,lambda2=1.0,lambda3=1e-5, **kwargs,):
        super().__init__(*args, **kwargs,L2=lambda3)
        self.ed = 1.0  # bonus for demonstration
        self.ea = 0.001
        self.margin = 0.8
        self.lambda1 = lambda1  # n-step return
        self.lambda2 = lambda2  # supervised loss
        self.lambda3 = lambda3  # L2
        self.replay.e = 0

    def storeTransition(self, s, a, r, s_, done, isdemo):
        s = torch.Tensor(s)
        s_ = torch.Tensor(s_)
        error = self.calcError((s, a, r, s_, done))
        e = self.ed if isdemo else self.ea
        self.store((s, a, r, s_, done, isdemo), error + e)

    def JE(self, samples):
        loss = torch.tensor(0.0)
        for s, a, *_, isdemo in samples:
            if not isdemo:
                continue
            QE = self.net(s)[a[0]]
            Q = max(self.net(s))
            Q = QE if Q + self.margin < QE else Q
            loss += self.lambda2 * (Q - QE)
        return loss / self.mbsize

    def update(self):
        self.opt.zero_grad()

        samples, idxs, IS = self.sample()
        if self.noisy:
            self.net.sample()  # for choosing action
        maxA = [self.findMaxA(s[3]) for s in samples]
        s, a, *_, isdemo = zip(*samples)
        s = torch.stack(s)
        if self.noisy:
            self.net.sample()  # for prediction
            self.net2.sample()  # for estimating Q
        predict = [self.net(s[i])[a[i][0]] for i in range(self.mbsize)]
        look_ahead = [r if done else r + self.gamma * self.net2(s_)[maxA[i][0]] for i, (s, a, r, s_, done, isdemo) in
                      enumerate(samples)]
        target = torch.Tensor(look_ahead)

        errors, ls = self.loss(predict, target, IS)
        if self.noisy:
            self.net.sample()
        ls += self.JE(samples)
        ls.backward()
        for i in range(self.mbsize):
            e = self.ed if isdemo[i] else self.ea
            self.replay.update(idxs[i], errors[i] + e)

        self.opt.step()
        if self.c >= self.C:
            self.c = 0
            self.net2.load_state_dict(self.net.state_dict())
            self.net2.eval()
        else:
            self.c += 1
