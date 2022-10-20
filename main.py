import torch
from torch import nn
from torch.nn import functional as fun
from torch import optim
import torchvision
from plot import xian, tu, chuli

size = 64  # 一次性导入64个图片
train = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                                               transform=torchvision.transforms.Compose(
                                                                          [torchvision.transforms.ToTensor(),
                                                                           torchvision.transforms.Normalize((0.1307,), (
                                                                               0.3081,))])), batch_size=size,
                                    shuffle=True)
test = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data2', train=False, download=True,
                                                              transform=torchvision.transforms.Compose(
                                                                         [torchvision.transforms.ToTensor(),
                                                                          torchvision.transforms.Normalize((0.1307,), (
                                                                              0.3081,))])), batch_size=size,
                                   shuffle=False)


x, y = next(iter(train))
class Wang(nn.Module):

    def __init__(self):
        # xw+b
        super(Wang, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # 输入784=28*28
        self.fc2 = nn.Linear(256, 64)  # 256和64为经验值
        self.fc3 = nn.Linear(64, 10)  # 输出10

    def forward(self, x):
        # x:[b,1,28,28]
        # h1=relu(xw1+b1)
        # h2=relu(xw2+b2)
        # h3=xw3+b3
        x = fun.relu(self.fc1(x))
        x = fun.relu(self.fc2(x))
        x = self.fc3(x)
        return x


w = Wang()
optimizer = optim.SGD(w.parameters(), lr=0.01, momentum=0.9)
trainloss = []

for each in range(5):
    for x_id, (x, y) in enumerate(train):
        # x:[b,1,28,28] y:[64] [b,784]->[b,10]
        x = x.view(x.size(0), 784)
        loss = fun.mse_loss(w(x), chuli(y))  # 计算误差，输出值和真实值的误差
        optimizer.zero_grad()  # 得到梯度
        loss.backward()  # w'=w-lr*grad
        optimizer.step()  # 不断训练得到最佳的w和b

        trainloss.append(loss.item())
        if x_id % 10 == 0:
            print(each, x_id, loss.item())

xian(trainloss)

num_correct = 0
for x, y in test:
    x = x.view(x.size(0), 784)
    predict = w(x).argmax(dim=1)  # 最大值所在的索引，预测值的数量
    correct = predict.eq(y).sum().float().item()  # 正确值的数量
    num_correct += correct
num = len(test.dataset)
print("准确率为:", num_correct / num)

x, y = next(iter(test))
predict = w(x.view(x.size(0), 784)).argmax(dim=1)
tu(x, predict, "tu wei")
