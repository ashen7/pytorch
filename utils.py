import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 选择数据集和模型
DATASET = "pokemon"
MODEL_NAME = "vggnet"
TEST_MODE = "eval"     # eval: evaluate model，test: test model inference 

# 网络超参
RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224
BATCH_SIZE = 50
LEARNING_RATE = 0.01
MAX_EPOCH = 50
ITER_INTERVAL = 1
SAVE_MODEL_INTERVAL = 2

# GPU Device
DEVICE_ID = 1
DEVICE_ID_LIST = [0, 1, 2, 3]
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

