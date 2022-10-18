import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt

from utils import plot_image, plot_curve, one_hot

batch_size = 512  # 一次性导入512个图片

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                                                      transform=torchvision.transforms.Compose(
                                                                          [torchvision.transforms.ToTensor(),
                                                                           torchvision.transforms.Normalize((0.1307,), (
                                                                               0.3081,))])), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data2', train=False, download=True,
                                                                     transform=torchvision.transforms.Compose(
                                                                         [torchvision.transforms.ToTensor(),
                                                                          torchvision.transforms.Normalize((0.1307,), (
                                                                              0.3081,))])), batch_size=batch_size,
                                          shuffle=False)

x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image')


class Net(nn.Module):

    def __init__(self):
        # xw+b
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # 输入784=28*28
        self.fc2 = nn.Linear(256, 64)  # 256和64为经验值
        self.fc3 = nn.Linear(64, 10)  # 输出10

    def forward(self, x):
        # x:[b,1,28,28]
        # h1=relu(xw1+b1)
        # h2=relu(xw2+b2)
        # h3=xw3+b3
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
train_loss = []

for each in range(5):
    for x_id, (x, y) in enumerate(train_loader):
        # x:[b,1,28,28] y:[512] [b,784]->[b,10]
        x = x.view(x.size(0), 28 * 28)
        out = net(x)
        loss = F.mse_loss(out, one_hot(y))  # 计算误差，输出值和真实值的误差
        optimizer.zero_grad()  # 得到梯度
        loss.backward()  # w'=w-lr*grad
        optimizer.step()  # 不断训练得到最佳的w和b

        train_loss.append(loss.item())
        if x_id % 10 == 0:
            print(each, x_id, loss.item())

plot_curve(train_loss)

num_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 784)
    predict = net(x).argmax(dim=1)  # 最大值所在的索引，预测值的数量
    correct = predict.eq(y).sum().float().item()  # 正确值的数量
    num_correct += correct
num = len(test_loader.dataset)
print("准确率为:", num_correct / num)

x, y = next(iter(test_loader))
predict = net(x.view(x.size(0), 784)).argmax(dim=1)
plot_image(x, predict, "image")
