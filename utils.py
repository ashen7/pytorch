import os
import warnings

import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from vggnet import VGGNet
from mobilenet_v3 import MobileNetV3_Large, MobileNetV3_Small
warnings.filterwarnings("ignore")

# 选择数据集和模型
DATASET = "pokemon"
MODEL_NAME = "vggnet"
TEST_MODE = "eval"       # eval: evaluate model，test: test model inference 
USE_PRETRAIN_MODEL = False

# 网络超参
RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224
BATCH_SIZE = 100
LEARNING_RATE = 0.01
MAX_EPOCH = 24
ITER_INTERVAL = 1
SAVE_MODEL_INTERVAL = 2

# GPU Device
DEVICE_ID = 1
DEVICE_ID_LIST = [0, 1, 2, 3]
USE_MULTIGPU = True

# 数据集
NUM_CORES = 6
DATASETS_DIR = os.path.join(os.path.abspath('.'), 'datasets')
CIFAR10_DIR = os.path.join(DATASETS_DIR, 'cifar-10-batches-py')
POKEMON_DIR = os.path.join(DATASETS_DIR, 'pokemon')

# loss和acc
TRAIN_LOSS_LIST = list()
VAL_ACC_LIST = list()
TEST_ACC_LIST = list()

# 导入模型
def load_model(model_name, use_pretrain_model, device, device_id = 0, use_multigpu = False, device_id_list = []):
    model = None

    if model_name == "vggnet":
        if use_pretrain_model:
            model = torchvision.models.vgg16(pretrained=True)
        else:
            model = VGGNet()
    elif model_name == "resnet18":
        pass
    elif model_name == "resnet50":
        pass
    elif model_name == "mobilenet_v3":
        model = MobileNetV3()
    elif model_name == "eff":
        pass
    else:
        print("暂不支持该模型!")
        exit(0)

    # 冻结前面的参数 只训练最后的全连接层
    if use_pretrain_model:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Sequential(
                    nn.Linear(100, 256), 
                    nn.ReLU(), 
                    nn.Dropout(0.4), 
                    nn.Linear(256, 5),
                    nn.LogSoftmax(dim=1)
                    )
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('模型参数总数：', total_params, ', 可训练参数总数：', total_trainable_params)

    # 用GPU运行
    if torch.cuda.device_count() > 1:
        if use_multigpu:
            model = nn.DataParallel(model, device_ids=device_id_list)
            device = torch.device("cuda:{}".format(device_id_list[0]))
        else:
            #os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
            device = torch.device("cuda:{}".format(device_id))
    model = model.to(device)

    return model

def plot_loss():
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(0, len(TRAIN_LOSS_LIST)), TRAIN_LOSS_LIST, 'g', label='loss')
    plt.legend(loc='best')
    plt.show()
    plt.savefig('loss.jpg')

def plot_acc():
    plt.title('Validation/Test Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('val/test acc')
    plt.plot(range(0, len(VAL_ACC_LIST)), VAL_ACC_LIST, 'g', label='val acc')
    plt.plot(range(0, len(TEST_ACC_LIST)), TEST_ACC_LIST, 'r', label='test acc')
    plt.legend(['validation accuracy', 'test accuracy'], loc='best')
    plt.show()
    plt.savefig('accuracy.jpg')

