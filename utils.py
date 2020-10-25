import os
import warnings

import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from backbone.vggnet import VGGNet
from backbone.mobilenet_v3 import MobileNetV3, MobileNetV3_Small
warnings.filterwarnings("ignore")

# 选择数据集和模型
DATASET = "pokemon"
MODEL_NAME = "mobilenet_v3"
TEST_MODE = "eval"       # eval: evaluate model，test: test model inference 
USE_PRETRAIN_MODEL = False
USE_MULTILABEL = True

# 网络超参
IMAGE_SIZE = 224
RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MAX_EPOCH = 20
ITER_INTERVAL = 1
SAVE_MODEL_INTERVAL = 2
if DATASET == "cifar10":
    IMAGE_SIZE = 32
INPUT_SIZE = int (512 * (4 + (IMAGE_SIZE / 32) - 1) * (4 + (IMAGE_SIZE / 32) - 1))

# GPU Device
DEVICE_ID = 1
DEVICE_ID_LIST = [1, 2, 3]
USE_MULTIGPU = False

# 数据集
NUM_CORES = 6
DATASETS_DIR = os.path.join(os.path.abspath('.'), 'datasets')
CIFAR10_DIR = os.path.join(DATASETS_DIR, 'cifar-10-batches-py')
POKEMON_DIR = os.path.join(DATASETS_DIR, 'pokemon')

# loss和acc
train_loss_list = list()
val_acc_list = list()
test_acc_list = list()

# 导入模型
def load_model(model_name, use_pretrain_model, use_multilabel, classes, input_size = None):
    model = None
    num_classes = None
    if use_multilabel:
        num_classes = [len(classes[0]), len(classes[1])]
    else:
        num_classes = len(classes)

    if model_name == "vggnet":
        if use_pretrain_model:
            model = torchvision.models.vgg16(pretrained=True)
        else:
            model = VGGNet(num_classes, input_size)
    elif model_name == "resnet18":
        pass
    elif model_name == "resnet50":
        pass
    elif model_name == "mobilenet_v3":
        model = MobileNetV3(num_classes, use_multilabel)
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
    print(model)

    return model

def plot_loss():
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(0, len(train_loss_list)), train_loss_list, 'g', label='loss')
    plt.legend(loc='best')
    plt.show()
    plt.savefig('results/loss.jpg')

def plot_acc():
    plt.title('Validation/Test Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('val/test acc')
    plt.plot(range(0, len(val_acc_list)), val_acc_list, 'g', label='val acc')
    plt.plot(range(0, len(test_acc_list)), test_acc_list, 'r', label='test acc')
    plt.legend(['validation accuracy', 'test accuracy'], loc='best')
    plt.show()
    plt.savefig('results/accuracy.jpg')

