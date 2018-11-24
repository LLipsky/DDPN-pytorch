import torch
x=torch.tensor([1,2,3])#axis=0代表列；axis=1代表行

print(x)
print(x.repeat(4,2))#猜测前面的4代表的是行repeat的次数；后者的2代表的是列repeat的次数。
print(x.repeat(4,2).size())

print(x.repeat(3,2))
print(x.repeat(3,2).size())

print(x.repeat(3,1))
print(x.repeat(3,1).size())

print(x.repeat(4,3))

print(x.repeat(4,2,1))

print('hahahh')
print(x.repeat(4,1))
print(x.repeat(4))
print(x.repeat(1,4))

import numpy as np
a=np.array([1,2,3])
print(np.tile(a,2).shape)
print(x.repeat(2).size())
