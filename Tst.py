from torch          import nn, randn, float32, zeros, matmul, einsum, FloatTensor, Tensor
from torch.autograd import Variable
import torch
import numpy as np

from Modified_MAMAL_for_MCANC import bulidng_data_by_DelayLine, Loss_Function_maml

# Fx =randn((4,4,4,512), dtype = float32)
# Wc = randn((4,4,512),  dtype = float32)

# A = einsum('rsen,rsn->rse', Fx, Wc)
# C = einsum('...se ->e',A)
# D = sum(C)**2

# print(A.shape)
# print(C.shape)
# print(D)

# E = FloatTensor([1, 2, 3, 4])

# F = einsum('i,i->',E, E)
# print(F)

# A1 = randn(3,2,3,4)
# B1 = randn(4)

# C1 = einsum('...e,...e ->...', A1, B1)

# print(f'C1 shape is {C1.shape}')

A = zeros((1,1,2,4), dtype= float32)
A[0,0,0] = FloatTensor([1, 2, 3, 4])
A[0,0,1] = FloatTensor([1, 2, 3, 4])

# print(A.shape)

# C=bulidng_data_by_DelayLine(4, A)

# print(C)

# x = torch.randn(1,2, 3)
# print(x.shape)
# y = x.permute(1, 0, 2)
# print(y.shape)

Fx        = zeros((1,1,1,4), dtype=float32)
Fx[0,0,0] = FloatTensor([1, 2, 3, 4])

Wc_np      = np.zeros((1,1,4), dtype=float)
Wc_np[0,0] = [1, 2, 3, 4]
Wc_tensor  = Tensor(Wc_np)
Wc         = Variable(Wc_tensor.detach(), requires_grad=True)


dr = FloatTensor([3]) 

def Adaptive_filtering(Weights, Fx):
    
    anti_noise_elements = einsum('...rsen,...rsn ->...rse', Fx, Weights)
    anti_noise          = einsum('...rse->...e', anti_noise_elements)
    
    return anti_noise


anti_noise = Adaptive_filtering(Wc, Fx)

error          = dr-anti_noise
Loss_1         = einsum('i,i->', error, error)
Loss_1.backward()


print(Wc.grad.detach())

gamma = FloatTensor([0.1, 0.01, 0.001, 0.0002])
A      = zeros((4,2), dtype=float32)
A[0,:] = FloatTensor([1, 1])
A[1,:] = FloatTensor([2, 2])
A[2,:] = FloatTensor([3, 3])
A[3,:] = FloatTensor([4, 4])
B      = zeros((2,4), dtype=float32) 
d1 = Loss_Function_maml(A,B,gamma)
print(d1)

c = [3, 4, 5]
print(c[-1])

print(f'There are {torch.cuda.device_count()} GPU availabel in the computer !!!')

dd1 = np.ones([3,4])

print(dd1.reshape(1,-1).shape)

tinydict = {'a': 1, 'b': 2, 'c': 3, 'd':4}

keys = list(tinydict.keys())
print(list(tinydict.keys()))
# print(tinydict.values().shape)

print(3/2)