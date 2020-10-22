import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        # 继承了nn的Module类 内部封装了autograd 这个类的成员都是requires_grad=True 自动求导的
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积1 后接 relu1
        x = F.relu(self.conv1(x))
        # 最大池化1
        x = F.max_pool2d(x, (2, 2))
        # 卷积2 后接 relu2
        x = F.relu(self.conv2(x))
        # 最大池化2
        x = F.max_pool2d(x, (2, 2))
        # view就是reshape
        single_x_size = 1
        hwc_list = x.size()[1:]
        for hwc in hwc_list:
            single_x_size *= hwc
        x = x.view(-1, single_x_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def main():
    net = Net()
    # 权值参数
    params = list(net.parameters())
    #n c h w
    x = torch.randn(1, 1, 32, 32)
    out = net(x)
    print(params)
    # 得到前向计算结果
    print(out)
    # 将所有参数的梯度缓存清零 进行随机梯度下降反向传播
    net.zero_grad()
    out.backward(torch.randn(1, 10))

if __name__ == '__main__':
    main()