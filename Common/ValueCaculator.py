import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc2 = nn.Linear(40, 3)

    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# (s,a) => Q(s,a)
class ValueCalculator1:
    def __init__(self, Net, actionFinder):
        self.predictNet = Net()
        self.targetNet = Net()
        self.actionFinder = actionFinder
        self.updateTargetNet()
        # (state => Action :List[List])

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
            # [[10.2],[5.3]] => [10.2,5.3]

    def sortedA(self, state):
        # return sorted action
        net = self.predictNet
        net.eval()
        A = self.actionFinder(state)
        Q = self.calcQ(net, state, A)
        A = [a for q,a in sorted(zip(Q, A),reverse=True)]
        net.train()
        return A

    def updateTargetNet(self):
        self.targetNet.load_state_dict(self.predictNet.state_dict())
        self.targetNet.eval()


# s => Q(s,a1), Q(s,a2)...
class ValueCalculator2:
    def __init__(self, Net):
        self.predictNet = Net()
        self.targetNet = Net()
        *_, last = self.predictNet.children()
        self.A = list(range(last.out_features))
        self.updateTargetNet()

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

    def sortedA(self, state):
        net = self.predictNet
        net.eval()
        Q = self.calcQ(net, state, self.A)
        A = [[a] for q,a in sorted(zip(Q,self.A),reverse=True)]
        net.train()
        return A

    def updateTargetNet(self):
        self.targetNet.load_state_dict(self.predictNet.state_dict())
        self.targetNet.eval()

if __name__ == "__main__":
    actionFinder = lambda x:[[0],[1],[2]]
    v1 = ValueCalculator1(Net,actionFinder)
    v2 = ValueCalculator2(Net2)
    s = torch.Tensor([1,2,3,4])
    print(v1.calcQ(v1.predictNet,s,actionFinder(s)))
    print(v1.sortedA(s))
    print(v2.calcQ(v2.predictNet,s,v2.A))
    print(v2.sortedA(s))
