import torch
import torchvision
import torchvision.transforms as transforms

from pokemon import Pokemon
from utils import *

def get_cifar10_training_dataset(batch_size):
    # 训练 数据增强
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),   #随机水平翻转
        transforms.RandomGrayscale(),        #随机灰度缩放(调整图片亮度)
        transforms.ToTensor(),               #numpy to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #归一化 均值 标准差
    ])

    # torchvision提供数据集
    train_dataset = torchvision.datasets.CIFAR10(root=DATASETS_DIR, train=True,
                                                 download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=NUM_CORES)
    print('Successfully load CIFAR10 training dataset')

    return train_loader

def get_cifar10_test_dataset(batch_size):
    # 测试
    transform_test = transforms.Compose([
        transforms.ToTensor(),  # numpy to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化 均值 标准差
    ])

    test_dataset = torchvision.datasets.CIFAR10(root=DATASETS_DIR, train=False,
                                                download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=NUM_CORES)
    print('Successfully load CIFAR10 test dataset')

    return test_loader

def get_pokemon_training_dataset(batch_size):
    train_dataset = Pokemon(POKEMON_DIR, RESIZE_HEIGHT, RESIZE_WIDTH, mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=NUM_CORES)
    print('Successfully load pokemon training dataset')

    return train_loader

def get_pokemon_test_dataset(batch_size):
    test_dataset = Pokemon(POKEMON_DIR, RESIZE_HEIGHT, RESIZE_WIDTH, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=NUM_CORES)
    print('Successfully load pokemon test dataset')

    return test_loader

def main():
    #dataset = get_pokemon_training_dataset(BATCH_SIZE)
    dataset = get_cifar10_training_dataset(BATCH_SIZE)
    for (batch_sample, batch_label) in dataset:
        print(batch_sample.shape, batch_label.shape)
        exit(0)

if __name__ == '__main__':
    main()
