import torch
from matplotlib import pyplot as p

def xian(data):
    p.plot(range(len(data)), data, color='green')
    p.legend(['zhi'], loc='upper right')
    p.xlabel('bu zhou')
    p.ylabel('zhi')
    p.show()


def tu(img, label, name):
    for i in range(6):
        p.subplot(2, 3, i + 1)
        p.tight_layout()
        p.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray_r', interpolation='none')
        p.title("{}:{}".format(name, label[i].item()))
        p.xticks([])
        p.yticks([])
    p.show()


def chuli(lable, depth=10):
    output = torch.zeros(lable.size(0), depth)
    idx = torch.LongTensor(lable).view(-1, 1)
    output.scatter_(dim=1, index=idx, value=1)
    return output
