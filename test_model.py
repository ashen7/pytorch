import os
import torch

from vggnet import VGGNet
from mobilenet_v3 import MobileNetV3_Large, MobileNetV3_Small
from data_preprocess import *
from utils import *

def test(model, device, model_path, get_test_dataset, batch_size = 100):
    test_loader = get_test_dataset(batch_size)
    test_batch_num = len(test_loader)
    test_acc = 0

    # 加载模型
    if not os.path.exists(model_path):
        pass
    else:
        # 导入模型
        pretrain_model = torch.load(model_path)
        # 恢复网络的参数
        model.load_state_dict(pretrain_model['model_state_dict'])
        print('Successfully Load {} Model'.format(model_path))

    # train时的BN作用和test不一样
    model.eval()
    # 前向计算 不用更新梯度
    with torch.no_grad():
        for test_data in test_loader:
            acc = 0
            test_batch_sample, test_batch_label = test_data
            # 得到测试数据 并传入cuda 默认是cpu to函数转为gpu
            test_batch_sample, test_batch_label = test_batch_sample.to(device), test_batch_label.to(device)
            output = model.forward(test_batch_sample)
            # data得到tensor转成python数组(用于数组)
            _, predict_output = torch.max(output.data, 1)
            acc = (predict_output == test_batch_label).sum().item()
            print('current batch acc is {}%'.format(acc))
            test_acc += acc

    print('test accuracy is: {}%'.format(test_acc / test_batch_num))

if __name__ == '__main__':
    get_training_dataset = None
    get_test_dataset = None
    model = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.getcwd() + "/models/{}.pth".format(MODEL_NAME)

    # 选择数据集
    if DATASET == "cifar10":
        get_test_dataset = get_cifar10_test_dataset
    elif DATASET == "pokemon":
        pass

    # 构建网络
    if MODEL_NAME == "vggnet":
        model = VGGNet()
    elif MODEL_NAME == "mobilenet_v3":
        model = MobileNetV3_Large()

    # 用GPU运行
    model = model.to(device)
    test(model, device, model_path, get_test_dataset, BATCH_SIZE)