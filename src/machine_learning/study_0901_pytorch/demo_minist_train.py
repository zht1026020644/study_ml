import torch
from torch import nn
from torch.nn import functional as F  # 函数包
from torch import optim # 优化器
import torchvision
from matplotlib import pyplot as plt
from utils import plot_image, plot_curve, one_hot

batchsize = 512  # batchsize 一次加载图片的数量
# step1 load dataset
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
# python 内置next(iter)函数，迭代器
# x, y = next(iter(train_loader))
# print(x.shape, y.shape, x.min(), y.min())
# plot_image(x,y,'image sample')

## model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # xw+b
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self,x):
        # x:[b,1,28,28]
        # h1= relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3 这里可以加softmax函数进行输出
        x = self.fc3(x)
        return x
# 训练网络
net = Net()
optimizer= optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
# 保存loss的值
train_loss = []
for epoch in range(3):
    for batch_idx, (x,y) in enumerate(train_loader):
        # x: [b,1,28,28], y[512]
        # [b,1,28,28] => [b,784]  把特征打横
        x = x.view(x.size(0),28*28)
        # =>[b,10]
        out = net(x)
        # [b,10]
        y_onehot = one_hot(y)
        # loss = mse(y,one_hot)
        loss = F.mse_loss(out, y_onehot)
        # 清零梯度
        optimizer.zero_grad()
        # 求导
        loss.backward()
        # w' = w- lr*grad 计算梯度
        optimizer.step()

        train_loss.append(loss.item())
        # if batch_idx % 10 == 0:
        #     print(epoch,batch_idx,loss.item(),loss)
# get optimal [w1,b1,w2,b2,w3,b3]
plot_curve(train_loss)

# 测试网络
total_correct = 0
for x,y in test_loader:
    # x: [b,1,28,28], y[512]
    # [b,1,28,28] => [b,784]  把特征打平
    x = x.view(x.size(0),28*28)
    out = net(x)
    # out:[b,10]
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct
total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:' ,acc)

x,y  = next(iter(test_loader))
out = net(x.view(x.size(0),28*28))
pred = out.argmax(dim=1)
plot_image(x,pred,'test')