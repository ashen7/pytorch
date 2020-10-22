import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision

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
train_loss_list = list()
val_acc_list = list()
test_acc_list = list()

# 选择数据集
def create_dataset():
    get_training_dataset, get_test_dataset = None, None
    if DATASET == "cifar10":
        get_training_dataset = get_cifar10_training_dataset
        get_test_dataset = get_cifar10_test_dataset
    elif DATASET == "pokemon":
        get_training_dataset = get_pokemon_training_dataset
        get_test_dataset = get_pokemon_test_dataset
    else:
        print("暂不支持该数据集!")
        exit(0)

    return get_training_dataset, get_test_dataset

# 构建网络
def create_model():
    if MODEL_NAME == "vggnet":
        if USE_PRETRAIN_MODEL:
            model = torchvision.models.vgg16(pretrained=True)
        else:
            model = VGGNet()
    elif MODEL_NAME == "mobilenet_v3":
        # model = MobileNetV3()
        pass
    else:
        print("暂不支持该模型!")
        exit(0)

    # 冻结前面的参数 只训练最后的全连接层
    if USE_PRETRAIN_MODEL:
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
        if USE_MULTIGPU:
            model = nn.DataParallel(model, device_ids=DEVICE_ID_LIST)
        else:
            #os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
            device = torch.device("cuda:{}".format(DEVICE_ID))
    model = model.to(device)

    return model

def plot_loss():
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(0, len(train_loss_list)), train_loss_list, 'g', label='loss')
    plt.legend(loc='best')
    plt.show()
    plt.savefig('loss.jpg')

def plot_acc():
    plt.title('Validation/Test Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('val/test acc')
    plt.plot(range(0, len(val_acc_list)), val_acc_list, 'g', label='val acc')
    plt.plot(range(0, len(test_acc_list)), test_acc_list, 'r', label='test acc')
    plt.legend(['validation accuracy', 'test accuracy'], loc='best')
    plt.show()
    plt.savefig('accuracy.jpg')

