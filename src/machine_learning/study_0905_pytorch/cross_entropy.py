import torch
from torch.nn import functional as F

x = torch.randn(1, 784)
w = torch.randn(10, 784)

logits = x @ w.t()

pred = F.softmax(logits, dim=1)

pred_log = torch.log(pred)

# 下面这两个函数结果相同
# 这个函数将softmax 函数和log打包在一起了
a=F.cross_entropy(logits, torch.tensor([3]))
print(a)
b = F.nll_loss(pred_log, torch.tensor([3]))
print(b)
