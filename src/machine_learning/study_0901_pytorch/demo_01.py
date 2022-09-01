import torch
import time
from torch import autograd

# a = torch.randn(10000, 1000)
# b = torch.randn(1000, 200)
# t0 = time.time()
# c = torch.matmul(a, b)
# t1 = time.time()
# print(a.device, t1 - t0, c.norm(2))
x = torch.tensor(1.)
print(x)
print('*' * 10)
a = torch.tensor(1., requires_grad=True)  # requires_grad = True 需要求解梯度
print(a)
print('*' * 10)
b = torch.tensor(2., requires_grad=True)
print(b)
print('*' * 10)
c = torch.tensor(3., requires_grad=True)
print(c)
print('*' * 10)

y = a ** 2 * x + x * b + c

print('before:', a.grad, b.grad, c.grad)
grads = autograd.grad(y, [a, b, c])
print('after', grads[0], grads[1], grads[2])
