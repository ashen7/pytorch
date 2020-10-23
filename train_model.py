import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
#import visdom

from data_preprocess import get_training_dataset, get_test_dataset
from utils import *

def train_model(model, device, model_path, train_loader, test_loader,
                batch_size = 100, learning_rate = 0.001, max_epoch = 20, momentum = 0.9):
    global TRAIN_LOSS_LIST
    global VAL_ACC_LIST
    global TEST_ACC_LIST
    train_total_samples = len(train_loader)
    train_batch_num = int(0.8 * train_total_samples)
    val_batch_num = int(0.2 * train_total_samples)

    init_epoch = 0
    criterion = None
    # 优化方法 选择SGD 因为模型每层使用了BN 所以可以使用大一点的学习率 加快收敛
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)

    # 可视化训练
    # viz = visdom.Visdom()
    # viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    # viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

    # 加载模型
    if os.path.exists(model_path) and USE_PRETRAIN_MODEL is not True:
        # 导入模型
        pretrain_model = torch.load(model_path, map_location='cpu')
        # 恢复网络的参数 得到上次训练的epoch 和loss
        init_epoch = pretrain_model['epoch']
        criterion = pretrain_model['criterion']
        model.load_state_dict(pretrain_model['model_state_dict'])
        optimizer.load_state_dict(pretrain_model['optimizer_state_dict'])
        print('Successfully Load {} Model'.format(model_path))
    else:
        # 定义交叉熵loss 包含softmax和loss
        criterion = nn.CrossEntropyLoss()

    # train时的BN作用和test不一样
    model.train()
    # BN Dropout层按训练时的来
    for epoch in range(init_epoch, max_epoch):
        begin = time.time()
        train_loss = 0
        train_acc = 0
        val_acc = 0
        global_step = 0
        exp_lr_scheduler.step()  

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
                    _, predict_output = torch.max(output.data, dim=1)
                    val_acc += (predict_output == val_batch_label).sum().item()
                    if (batch_idx + 1) == train_total_samples:
                        print('=====================Epoch {} validation accuracy is: {}%, spend time: {}s====================='.format(epoch + 1, val_acc / val_batch_num, time.time() - begin))
                        VAL_ACC_LIST.append(val_acc / val_batch_num)
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
                _, predict_output = torch.max(output.data, dim=1)
                train_acc += (predict_output == train_batch_label).sum().item()
                if (batch_idx + 1) % ITER_INTERVAL == 0:
                    print('train loss : {}, train acc : {}%'.format(train_loss / ITER_INTERVAL, train_acc / ITER_INTERVAL))
                    TRAIN_LOSS_LIST.append(train_loss / ITER_INTERVAL)
                    train_loss = 0
                    train_acc = 0

        # 这里一轮迭代完成 每迭代2轮保存一次模型 并测试一次
        if (epoch + 1) % SAVE_MODEL_INTERVAL == 0:
            # 得到测试集
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
                    _, predict_output = torch.max(output.data, dim=1)
                    test_acc += (predict_output == test_batch_label).sum().item()
            print('=====================Epoch {} test accuracy is: {}%====================='.format(epoch + 1, test_acc / test_batch_num))
            TEST_ACC_LIST.append(test_acc / test_batch_num)
        
            #if torch.cuda.device_count() > 1 and USE_MULTIGPU:
            if isinstance(model, torch.nn.DataParallel):
                torch.save({
                    'epoch': epoch,
                    #'model_state_dict': model.module.state_dict(),
                    'model_state_dict': model.state_dict(),
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
    model_name = MODEL_NAME
    use_pretrain_model = USE_PRETRAIN_MODEL
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_id = DEVICE_ID
    use_multigpu = USE_MULTIGPU
    device_id_list = DEVICE_ID_LIST
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    max_epoch = MAX_EPOCH
    dataset = DATASET
    model_path = os.getcwd() + "/models/{}_{}.pth".format(model_name, dataset)

    # 得到数据集
    train_loader = get_training_dataset(batch_size)
    test_loader, classes = get_test_dataset(batch_size)
    # 构建网络 
    model = load_model(model_name, use_pretrain_model, device, device_id, use_multigpu, device_id_list) 
    print(model)

    # 训练模型
    train_model(model, device, model_path, train_loader, test_loader, 
                batch_size, learning_rate, max_epoch)
    plot_loss()
    plot_acc()

if __name__ == '__main__':
    main()
