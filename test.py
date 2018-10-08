import numpy as np
from struct import unpack
import Qnet
import torch
from trainWithMcts import testResult

'''
net = Qnet.Net3()
net.load_state_dict(torch.load("ddz_optimal_34.txt"))

data = net.toNumpy()
for name,layer in data.items():
    np.save(name,layer)

sa=np.arange(75)
s = torch.Tensor(sa[:60])
a = torch.Tensor(sa[60:])
print(sa)
print(net.fc1_s(s)+net.fc1_a(a))
'''


test = [[[-122.45939656328747, 37.796690447896445], [-122.45859061899071, 37.785810199890264], [-122.44198816647757, 37.786535549757346], [-122.43578239539256, 37.789920515803715], [-122.42828711343275, 37.77444638530603]]]
result = [str(coor).strip('[]') for coor in test[0]]
result = " | ".join(result)
print(result)