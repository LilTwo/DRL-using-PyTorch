import math

import torch
from torch.nn.parameter import Parameter
from torch import nn
import torch.nn.functional as F


def f(input):
    sign = torch.sign(input)
    return sign * (torch.sqrt(torch.abs(input)))


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,sig0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_sig = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_sig = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(sig0)
        self.dist = torch.distributions.Normal(0, 1)
        self.weight = None
        self.bias = None
        self.sample()

    def reset_parameters(self,sig0):
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_sig.data = self.weight_sig.data.zero_() + sig0 / self.weight_mu.shape[1]

        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_sig.data.zero_()
            self.bias_sig.data = self.bias_sig.data.zero_() + sig0 / self.weight_mu.shape[1]

    def sample(self, zero=1):
        size_in = self.in_features
        size_out = self.out_features
        noise_in = f(self.dist.sample((1, size_in))) * zero
        noise_out = f(self.dist.sample((1, size_out))) * zero
        self.weight = self.weight_mu + self.weight_sig * torch.mm(noise_out.t(), noise_in)
        self.bias = (self.bias_mu + self.bias_sig * noise_out).squeeze()

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def randomness(self):
        size_in = self.in_features
        size_out = self.out_features
        return torch.abs(self.bias_sig.data/self.bias_mu.data).numpy().sum()/size_out#+torch.abs(self.weight_sig.data/self.weight_mu.data).numpy().sum()/(size_in*size_out)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


if __name__ == "__main__":
    a = torch.Tensor([[1, -2, 3]])
    b = torch.Tensor([1, 2, 3])
    n = NoisyLinear(3, 100)

    print(n.bias_sig.data.zero_())
    print(n.weight_sig.data.zero_())
    print(n.randomness())
