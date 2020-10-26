import os
import glob
import time
import random
import csv

import torch
import torchvision
import torchvision.transforms as transforms
import visdom
from PIL import Image

from utils import *

class Pokemon(torch.utils.data.Dataset):
    def __init__(self, root, resize_h, resize_w, mode = None, use_multilabel = False):
        super(Pokemon, self).__init__()
        self.root = root
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.mode = mode
        self.use_multilabel = use_multilabel
        self.labels_dict = dict()
        count = 0

        # 数据增强
        self.train_transform = transforms.Compose([
            #lambda x: Image.open(x).convert('RGB'),          # 读图片解码并转为RGB
            transforms.Resize((int(self.resize_h * 1.25), int(self.resize_w * 1.25))),  # resize
            transforms.RandomRotation(15),                    # 随机旋转 设置旋转的度数小一些，否则会增加网络的学习难度
            transforms.RandomHorizontalFlip(),                # 随机水平翻转
            # transforms.ColorJitter(),                         # 颜色抖动
            transforms.CenterCrop(self.resize_h),             # 中心裁剪 此时：既旋转了又不至于导致图片变得比较的复杂
            transforms.ToTensor(),                            # numpy.ndarray to torch.tensor 并且 / 255到[0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准 归一化 每个通道的均值和方差
                                 std=[0.229, 0.224, 0.225])
        ])

        # 数据增强
        self.test_transform = transforms.Compose([
            transforms.Resize((int(self.resize_h * 1.25), int(self.resize_w * 1.25))),  # resize
            transforms.CenterCrop(self.resize_h),             # 中心裁剪 此时：既旋转了又不至于导致图片变得比较的复杂
            transforms.ToTensor(),                            # numpy.ndarray to torch.tensor 并且 / 255到[0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准 归一化 每个通道的均值和方差
                                 std=[0.229, 0.224, 0.225])
        ])

        if self.mode == None:
            return

        for classify_name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, classify_name)):
                continue
            # 保存在表中;将最长的映射作为最新的元素的label的值
            self.labels_dict[classify_name] = count
            count += 1

        # 加载文件
        if self.use_multilabel:
            self.samples, self.labels = self.load_csv('pokemon_multilabel.csv')
        else:
            self.samples, self.labels = self.load_csv('pokemon.csv')
        # 裁剪数据
        if mode == 'train':
            self.samples = self.samples[:int(0.8 * len(self.samples))]
            self.labels = self.labels[:int(0.8 * len(self.labels))]
        elif mode == 'test':
            self.samples = self.samples[int(0.8 * len(self.samples)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    # 导入图片
    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for classify_name in self.labels_dict.keys():
                images += glob.glob(os.path.join(self.root, classify_name, '*.png'))
                images += glob.glob(os.path.join(self.root, classify_name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, classify_name, '*.jpeg'))

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for image in images:
                    classify_name = image.split(os.sep)[-2]
                    label = self.labels_dict[classify_name]
                    if self.use_multilabel:
                        label1 = label
                        label2 = label
                        writer.writerow([image, label1, label2])
                    else:
                        writer.writerow([image, label])
                print("write into csv into :", filename)

        samples, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                sample, label = None, None
                if self.use_multilabel:
                    # 将label转码为int类型
                    sample, label1, label2 = row
                    label = [int(label1), int(label2)]
                else:
                    sample, label = row
                    label = int(label)
                samples.append(sample)
                labels.append(label)

        assert len(samples) == len(labels)
        return samples, labels

    # 返回数据的数量
    def __len__(self):
        return len(self.samples)

    # 返回一个sample，label
    def __getitem__(self, index):
        sample, label = self.samples[index], self.labels[index]
        sample = Image.open(sample).convert('RGB')  # 读图片解码RGB
        if self.mode == 'train':
            sample = self.train_transform(sample)
        else:
            sample = self.test_transform(sample)
        label = torch.tensor(label)

        return sample, label

    # 反归一化
    def denormalize(self, sample):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # unsqueeze(x)在第x维上添加一个维度
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        origin_sample = sample * std + mean
        return origin_sample
    
    # 图像增强
    def transform(self, image):
        return self.test_transform(image)

def main():
    # 验证工作
    viz = visdom.Visdom()

    resize_h, resize_w = 224, 224
    mode = 'train'
    pokemon_dataset = Pokemon(POKEMON_DIR, resize_h, resize_w, mode)
    # 可视化样本
    # sample, label = next(iter(pokemon_dataset))
    # viz.image(pokemon_dataset.denormalize(sample), win='sample', opts=dict(title='sample'))

    # 加载batch_size的数据
    loader = torch.utils.data.DataLoader(pokemon_dataset, batch_size=32, shuffle=True, num_workers=NUM_CORES)
    for sample, label in loader:
        viz.images(pokemon_dataset.denormalize(sample), nrow=8, win='samples', opts=dict(title='samples'))
        viz.text(str(label.numpy()), win='labels', opts=dict(title='labels'))
        time.sleep(10)

if __name__ == '__main__':
    main()


