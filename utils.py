import os
import warnings
import argparse

import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from backbone.vggnet import VGGNet
from backbone.mobilenet_v3 import MobileNetV3, MobileNetV3_Small
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()

    # 选择数据集和模型
    parser.add_argument('-d', '--dataset',      type=str, default='pokemon', 
                        help='select dataset')
    parser.add_argument('-m', '--model_name',   type=str, default='mobilenet_v3', 
                        help='select model')
    parser.add_argument('-pretrain', '--use_pretrain_model', action='store_true', default=False, 
                        help='pretrained mdoel')
    parser.add_argument('-mlabel', '--use_multilabel',         action='store_true', default=False, 
                        help='single label or multi label')
    parser.add_argument('-t', '--test_mode',    type=str, default='eval',  
                        help='test mode(eval/test)')

    # GPU Device
    parser.add_argument('-id', '--device_id',  type=int, default=1, 
                        help='gpu device id')
    parser.add_argument('-id_list', '--device_id_list',  type=list, default=[1, 2, 3], 
                        help='gpu device id list')
    parser.add_argument('-mgpu', '--use_multigpu', action='store_true', default=False, 
                        help='single gpu or multi gpu')

    # 网络超参
    parser.add_argument('-s', '--image_size', type=int, default=224, 
                        help='image size')
    parser.add_argument('-rh', '--resize_h',  type=int, default=224, 
                        help='resize height')
    parser.add_argument('-rw', '--resize_w',  type=int, default=224, 
                        help='resize width')
    parser.add_argument('-b', '--batch_size', type=int, default=64, 
                        help='batch size')
    parser.add_argument('-e', '--max_epoch',  type=int, default=20, 
                        help='max epoch')
    parser.add_argument('-lr', '--learning_rate',           type=float, default=0.01, 
                        help='learning rate')
    parser.add_argument('-p', '--print_iter_interval',      type=int, default=1, 
                        help='print iter interval')
    parser.add_argument('-save', '--save_model_interval',   type=int, default=2, 
                        help='save model interval')

    # argparse
    args = parser.parse_args()

    return args

# 数据集
CPU_CORES = 6 
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

