import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
#import visdom

from vggnet import VGGNet
from mobilenet_v3 import MobileNetV3_Large, MobileNetV3_Small
from data_preprocess import *
from utils import *

def train_model(model, device, model_path, get_training_dataset, get_test_dataset,
                batch_size = 100, learning_rate = 0.001, max_epoch = 20):
    global val_acc_list
    global test_acc_list
    train_loader = get_training_dataset(batch_size=batch_size)
    train_total_samples = len(train_loader)
    train_batch_num = int(0.8 * train_total_samples)
    val_batch_num = int(0.2 * train_total_samples)

    init_epoch = 0
    # 优化方法 选择SGD 因为模型每层使用了BN 所以可以使用大一点的学习率 加快收敛
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 可视化训练
    # viz = visdom.Visdom()
    # viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    # viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

    # 加载模型
    if not os.path.exists(model_path):
        # 定义交叉熵loss 包含softmax和loss
        criterion = nn.CrossEntropyLoss()
    else:
        # 导入模型
        pretrain_model = torch.load(model_path, map_location='cpu')
        # 恢复网络的参数 得到上次训练的epoch 和loss
        model.load_state_dict(pretrain_model['model_state_dict'])
        optimizer.load_state_dict(pretrain_model['optimizer_state_dict'])
        init_epoch = pretrain_model['epoch']
        criterion = pretrain_model['criterion']
        print('Successfully Load {} Model'.format(model_path))

    # train时的BN作用和test不一样
    model.train()
    # BN Dropout层按训练时的来
    for epoch in range(init_epoch, max_epoch):
        begin = time.time()
        train_loss = 0
        train_acc = 0
        val_acc = 0
        global_step = 0

        for batch_idx, train_data in enumerate(train_loader, start=0):
            if batch_idx >= train_batch_num:
                # 不计算梯度
                with torch.no_grad():
                    # 验证
                    val_batch_sample, val_batch_label = train_data
                    # 得到验   证数据 并传入cuda 默认是cpu to函数转为gpu
                    val_batch_sample, val_batch_label = val_batch_sample.to(device), val_batch_label.to(device)
                    # 前向计算
                    output = model.forward(val_batch_sample)
                    # data得到tensor转成python数组(用于数组)
                    _, predict_output = torch.max(output.data, 1)
                    val_acc += (predict_output == val_batch_label).sum().item()
                    if (batch_idx + 1) == train_total_samples:
                        print('=====================Epoch {} validation accuracy is: {}%, spend time: {}s====================='.format(epoch + 1, val_acc / val_batch_num, time.time() - begin))
                        val_acc_list.append(val_acc / val_batch_num)
                        # viz.line([val_acc / ITER_INTERVAL], [global_step], win='val_acc', update='append')
                        val_acc = 0
            else:
                # 训练
                train_batch_sample, train_batch_label = train_data
                # 得到训练数据 并传入cuda 默认是cpu to函数转为gpu
                train_batch_sample, train_batch_label = train_batch_sample.to(device), train_batch_label.to(device)
                # 将参数梯度置0
                optimizer.zero_grad()
                # 前向计算 + 反向传播 + 权值更新
                output = model.forward(train_batch_sample)
                loss = criterion(output, train_batch_label)
                loss.backward()
                optimizer.step()

                # item得到tensor转成python浮点值(用于单个元素)
                train_loss += loss.item()
                # viz.line([loss.item()], [global_step], win='loss', update='append')
                global_step += 1
                # data得到tensor转成python数组(用于数组)
                _, predict_output = torch.max(output.data, 1)
                train_acc += (predict_output == train_batch_label).sum().item()
                if (batch_idx + 1) % ITER_INTERVAL == 0:
                    print('loss : {}, train acc : {}%'.format(train_loss / ITER_INTERVAL, train_acc / ITER_INTERVAL))
                    train_loss_list.append(train_loss / ITER_INTERVAL)
                    train_loss = 0
                    train_acc = 0

        # 这里一轮迭代完成 每迭代2轮保存一次模型 并测试一次
        if (epoch + 1) % SAVE_MODEL_INTERVAL == 0:
            # 得到测试集
            test_loader = get_test_dataset(batch_size)
            test_batch_num = len(test_loader)
            test_acc = 0
            # 不计算梯度
            with torch.no_grad():
                for test_data in test_loader:
                    test_batch_sample, test_batch_label = test_data
                    # 得到测试数据 并传入cuda 默认是cpu to函数转为gpu
                    test_batch_sample, test_batch_label = test_batch_sample.to(device), test_batch_label.to(device)
                    # 前向计算
                    output = model.forward(test_batch_sample)
                    # data得到tensor转成python数组(用于数组)
                    _, predict_output = torch.max(output.data, 1)
                    test_acc += (predict_output == test_batch_label).sum().item()
            print('=====================Epoch {} test accuracy is: {}%====================='.format(epoch + 1, test_acc / test_batch_num))
            test_acc_list.append(test_acc / test_batch_num)
        
            if torch.cuda.device_count() > 1 and USE_MULTIGPU:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'criterion': criterion
                }, model_path)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'criterion': criterion
                }, model_path)
            print('Save Model', model_path)

    print('Finished Training')

def main():
    torch.manual_seed(1234)
    get_training_dataset = None
    get_test_dataset = None
    model = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.getcwd() + "/models/{}_{}.pth".format(MODEL_NAME, DATASET)

    # 选择数据集
    if DATASET == "cifar10":
        get_training_dataset = get_cifar10_training_dataset
        get_test_dataset = get_cifar10_test_dataset
    elif DATASET == "pokemon":
        get_training_dataset = get_pokemon_training_dataset
        get_test_dataset = get_pokemon_test_dataset

    # 构建网络
    if MODEL_NAME == "vggnet":
        model = VGGNet()
    elif MODEL_NAME == "mobilenet_v3":
        # model = MobileNetV3()
        pass

    # 用GPU运行
    if torch.cuda.device_count() > 1:
        if USE_MULTIGPU:
            model = nn.DataParallel(model, device_ids=DEVICE_ID_LIST)
        else:
            #os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
            device = torch.device("cuda:{}".format(DEVICE_ID))
    model = model.to(device)
    train_model(model, device, model_path, get_training_dataset, get_test_dataset, BATCH_SIZE, LEARNING_RATE, MAX_EPOCH)
    plot_loss()
    plot_acc()

if __name__ == '__main__':
    main()
