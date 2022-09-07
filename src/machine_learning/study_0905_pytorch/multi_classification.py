import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
from visdom import Visdom

batchsize = 200
w1, b1 = torch.randn(200, 784, requires_grad=True), torch.zeros(200, requires_grad=True)
w2, b2 = torch.randn(200, 200, requires_grad=True), torch.zeros(200, requires_grad=True)
w3, b3 = torch.randn(10, 200, requires_grad=True), torch.zeros(10, requires_grad=True)

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                                                      transform=torchvision.transforms.Compose(
                                                                          [torchvision.transforms.ToTensor(),
                                                                           torchvision.transforms.Normalize((0.1307), (
                                                                               0.3081))])), batch_size=batchsize,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                                                     transform=torchvision.transforms.Compose(
                                                                         [torchvision.transforms.ToTensor(),
                                                                          torchvision.transforms.Normalize((0.1307), (
                                                                              0.3081))])), batch_size=batchsize,

                                          shuffle=False)

# 初始化
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)


# 网络
def forward(x):
    x = x @ w1.t() + b1
    x = F.relu(x)
    x = x @ w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    x = F.relu(x)
    return x


# train
learning_rate = 0.01
# 优化器  weight_decay = 0.01 加入二阶正则化 防止过拟合,momentum= 0.78加入动量 Adam优化器内置了momentum
optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate, weight_decay=0.01,momentum= 0.78)
# F.cross_entropy()功能相同
criteon = nn.CrossEntropyLoss()

# 使用visdom进行可视化
viz = Visdom()
# 初始化窗口
viz.line([0.], [0.], win="train loss", opts=dict(title='train_loss'))
epochs = 10
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        logits = forward(data)
        loss = criteon(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(epoch, batch_idx * len(data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx / len(train_loader),
                                                                         loss.item()))

test_loss = 0
correct = 0
for data, target in test_loader:
    data = data.view(-1, 28 * 28)
    logits = forward(data)
    loss = criteon(logits, target)
    test_loss += loss.item()
    pred = logits.data.max(1)[1]
    correct += pred.eq(target.data).sum()
test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss:{:.4f},Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
                                                                           100. * correct / len(test_loader.dataset)))
viz.line([test_loss], [epoch], win="train loss", update='append')
