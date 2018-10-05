import random
import numpy as np
if __package__:
    from .SumTree import SumTree
else:
    from SumTree import SumTree
import torch.nn as nn
import torch
import math

#MSE with importance sampling
class WeightedMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, weights):
        l = torch.tensor(0.0)
        errors = []
        for input, target, weight in zip(inputs, targets, weights):
            error = input - target
            l += error ** 2 * weight

        return l / weights.shape[0]


#the following code is from
#https://github.com/rlcode/per
class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.0
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, epoch=150):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, sample,error=None):
        if error is None:
            p = self.tree.tree[0] #max priority for new data
            if p == 0:
                p = 0.1
            else:
                p = self.tree.get(p*0.9)[1]
        else:
            p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


if __name__ == "__main__":
    print("memory test")
    m = Memory(10, 100)
    m.add(1, (3, 4))
    m.add(10, (5, 6))
    m.add(100, (7, 8))
    data, idx, _ = m.sample(2)
    print(m.sample(5))
    m.update(11, 0)
    print(m.sample(5))
