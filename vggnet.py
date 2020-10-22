import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGNet(nn.Module):
    def __init__(self):
        # 调用VGGNet的父类的构造函数
        super(VGGNet, self).__init__()

        # 输入通道3 输出通道特征图个数64 filter 3*3 步长1
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        # filter 2*2 步长2
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # BN 归一化到均值0 标准差1的正态分布 特征图数量64
        self.bn1 = nn.BatchNorm2d(64)
        # 激活函数 非线性变换
        self.relu1 = nn.ReLU()

        # 输入通道64 输出通道特征图个数128 filter 3*3 步长1
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        # 输入通道128 输出通道特征图个数128 filter 3*3 步长1
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # 输入通道128 输出通道特征图个数256 filter 3*3 步长1
        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        # 输入通道256 输出通道特征图个数512 filter 3*3 步长1
        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        # 全连接层  输入节点 输出节点
        self.fc14 = nn.Linear(512*5*5, 1024)
        self.dropout1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024, 10)

    # 前向计算
    def forward(self, input):
        vggnet = self.conv1(input)
        vggnet = self.conv2(vggnet)
        vggnet = self.pool1(vggnet)
        vggnet = self.bn1(vggnet)
        vggnet = self.relu1(vggnet)

        vggnet = self.conv3(vggnet)
        vggnet = self.conv4(vggnet)
        vggnet = self.pool2(vggnet)
        vggnet = self.bn2(vggnet)
        vggnet = self.relu2(vggnet)

        vggnet = self.conv5(vggnet)
        vggnet = self.conv6(vggnet)
        vggnet = self.conv7(vggnet)
        vggnet = self.pool3(vggnet)
        vggnet = self.bn3(vggnet)
        vggnet = self.relu3(vggnet)

        vggnet = self.conv8(vggnet)
        vggnet = self.conv9(vggnet)
        vggnet = self.conv10(vggnet)
        vggnet = self.pool4(vggnet)
        vggnet = self.bn4(vggnet)
        vggnet = self.relu4(vggnet)

        vggnet = self.conv11(vggnet)
        vggnet = self.conv12(vggnet)
        vggnet = self.conv13(vggnet)
        vggnet = self.pool5(vggnet)
        vggnet = self.bn5(vggnet)
        vggnet = self.relu5(vggnet)
        # print(vggnet.size())

        # view就是reshape
        vggnet = vggnet.view(-1, vggnet.shape[1] * vggnet.shape[2] * vggnet.shape[3])
        vggnet = F.relu(self.fc14(vggnet))
        vggnet = self.dropout1(vggnet)
        vggnet = F.relu(self.fc15(vggnet))
        vggnet = self.dropout2(vggnet)
        vggnet = self.fc16(vggnet)

        # 前向计算不用softmax 训练时Loss会调用 测试时不用
        return vggnet

if __name__ == '__main__':
    pass

