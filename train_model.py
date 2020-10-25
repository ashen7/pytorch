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
    global train_loss_list
    global val_acc_list
    global test_acc_list
    train_total_samples = len(train_loader)
    train_iter_num = int(0.8 * train_total_samples)
    val_iter_num = int(0.2 * train_total_samples)

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
    for epoch in range(init_epoch, max_epoch):
        begin = time.time()
        train_loss = 0
        train_acc = 0
        val_acc = 0
        val_iter = 0
        global_step = 0
        exp_lr_scheduler.step()

        for batch_idx, train_data in enumerate(train_loader, start=0):
            if batch_idx >= train_iter_num:
                # 不计算梯度
                with torch.no_grad():
                    # 验证
                    val_batch_sample, val_batch_label = train_data
                    # 得到验证数据 默认是cpu to转为gpu
                    val_batch_sample, val_batch_label = val_batch_sample.to(device), val_batch_label.to(device)
                    # 前向计算
                    output = model.forward(val_batch_sample)
                    # data得到tensor转成python数组(用于数组)
                    _, predict_output = torch.max(output.data, dim=1)
                    correct = (predict_output == val_batch_label).sum().item()
                    acc = (correct / len(predict_output))
                    val_acc += acc
                    val_iter += 1
            else:
                # 训练
                train_batch_sample, train_batch_label = train_data
                # 得到训练数据 默认是cpu to转为gpu
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
                correct = (predict_output == train_batch_label).sum().item()
                acc = (correct / len(predict_output))
                train_acc += acc
                if (batch_idx + 1) % ITER_INTERVAL == 0:
                    print('train loss : {}, train acc : {}%'.format(train_loss / ITER_INTERVAL, round(train_acc / ITER_INTERVAL * 100.0, 2)))
                    train_loss_list.append(train_loss / ITER_INTERVAL)
                    train_loss = 0
                    train_acc = 0
        print('=====================Epoch {} validation accuracy is: {}%, spend time: {}s====================='.format(epoch + 1, round(val_acc / val_iter * 100.0, 2), time.time() - begin))
        val_acc_list.append(val_acc / val_iter)
        # viz.line([val_acc / ITER_INTERVAL], [global_step], win='val_acc', update='append')
        val_acc = 0
        val_iter = 0

        # 这里一轮迭代完成 每迭代2轮保存一次模型 并测试一次
        if (epoch + 1) % SAVE_MODEL_INTERVAL == 0:
            # 得到测试集
            test_iter_num = len(test_loader)
            test_acc = 0
            test_iter = 0
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
                    correct = (predict_output == test_batch_label).sum().item()
                    acc = correct / len(predict_output)
                    test_acc += acc
                    test_iter += 1
            print('=====================Epoch {} test accuracy is: {}%====================='.format(epoch + 1, round(test_acc / test_iter * 100, 2)))
            test_acc_list.append(test_acc / test_iter)

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

def train_multitask_model(model, device, model_path, train_loader, test_loader,
                          batch_size=100, learning_rate=0.001, max_epoch=20, momentum=0.9):
    global train_loss_list
    global val_acc_list
    global test_acc_list
    train_total_samples = len(train_loader)
    train_iter_num = int(0.8 * train_total_samples)
    val_iter_num = int(0.2 * train_total_samples)

    init_epoch = 0
    criterion = None
    # 优化方法 选择SGD 因为模型每层使用了BN 所以可以使用大一点的学习率 加快收敛
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)

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
    for epoch in range(init_epoch, max_epoch):
        begin = time.time()
        train_loss = 0
        train_acc1 = 0
        train_acc2 = 0
        val_acc1 = 0
        val_acc2 = 0
        val_iter = 0
        exp_lr_scheduler.step()

        for batch_idx, train_data in enumerate(train_loader, start=0):
            if batch_idx >= train_iter_num:
                # 不计算梯度
                with torch.no_grad():
                    # 验证
                    val_batch_sample, val_batch_label = train_data
                    # 得到验证数据 默认是cpu to转为gpu
                    val_batch_sample, val_batch_label = val_batch_sample.to(device), val_batch_label.to(device)
                    # 前向计算
                    output = model.forward(val_batch_sample)
                    output1, output2 = output
                    label1, label2 = val_batch_label[:, 0], val_batch_label[:, 1]
                    _, predict_output1 = torch.max(output1.data, dim=1)
                    _, predict_output2 = torch.max(output2.data, dim=1)
                    correct1 = (predict_output1 == label1).sum().item()
                    correct2 = (predict_output2 == label2).sum().item()
                    acc1 = (correct1 / len(predict_output1))
                    acc2 = (correct2 / len(predict_output2))
                    val_acc1 += acc1
                    val_acc2 += acc2
                    val_iter += 1
            else:
                # 训练
                train_batch_sample, train_batch_label = train_data
                # 得到训练数据 默认是cpu to转为gpu
                train_batch_sample, train_batch_label = train_batch_sample.to(device), train_batch_label.to(device)
                # 将参数梯度置0
                optimizer.zero_grad()
                # 前向计算 + 反向传播 + 权值更新
                output = model.forward(train_batch_sample)
                loss = None
                output1, output2 = output
                label1, label2 = train_batch_label[:, 0], train_batch_label[:, 1]
                loss1 = criterion(output1, label1)
                loss2 = criterion(output2, label2)
                loss = loss1 + loss2
                # print(loss1, loss2, loss)
                loss.backward()
                optimizer.step()

                # item得到tensor转成python浮点值(用于单个元素)
                train_loss += loss.item()
                # data得到tensor转成python数组(用于数组)
                _, predict_output1 = torch.max(output1.data, dim=1)
                _, predict_output2 = torch.max(output2.data, dim=1)
                correct1 = (predict_output1 == label1).sum().item()
                correct2 = (predict_output2 == label2).sum().item()
                acc1 = (correct1 / len(predict_output1))
                acc2 = (correct2 / len(predict_output2))
                train_acc1 += acc1
                train_acc2 += acc2
                if (batch_idx + 1) % ITER_INTERVAL == 0:
                    print('train loss : {}, task1 train acc: {}%, task2 train acc: {}%'.format(train_loss / ITER_INTERVAL,
                                                                                          round(train_acc1 / ITER_INTERVAL * 100.0, 2),
                                                                                          round(train_acc2 / ITER_INTERVAL * 100.0, 2)))
                    train_loss_list.append(train_loss / ITER_INTERVAL)
                    train_loss = 0
                    train_acc1 = 0
                    train_acc2 = 0
        print('=====================Epoch {} task1 val acc: {}%, task2 val acc: {}%, spend time: {}s====================='.format(
              epoch + 1,
              round(val_acc1 / val_iter * 100.0, 2),
              round(val_acc2 / val_iter * 100.0, 2),
              time.time() - begin))
        val_acc1 = 0
        val_acc2 = 0
        val_iter = 0

        # 这里一轮迭代完成 每迭代2轮保存一次模型 并测试一次
        if (epoch + 1) % SAVE_MODEL_INTERVAL == 0:
            # 得到测试集
            test_iter_num = len(test_loader)
            test_acc1 = 0
            test_acc2 = 0
            test_iter = 0
            # 不计算梯度
            with torch.no_grad():
                for test_data in test_loader:
                    test_batch_sample, test_batch_label = test_data
                    # 得到测试数据 并传入cuda 默认是cpu to函数转为gpu
                    test_batch_sample, test_batch_label = test_batch_sample.to(device), test_batch_label.to(device)
                    # 前向计算
                    output = model.forward(test_batch_sample)
                    output1, output2 = output
                    label1, label2 = test_batch_label[:, 0], test_batch_label[:, 1]
                    _, predict_output1 = torch.max(output1.data, dim=1)
                    _, predict_output2 = torch.max(output2.data, dim=1)
                    correct1 = (predict_output1 == label1).sum().item()
                    correct2 = (predict_output2 == label2).sum().item()
                    acc1 = (correct1 / len(predict_output1))
                    acc2 = (correct2 / len(predict_output2))
                    test_acc1 += acc1
                    test_acc2 += acc2
                    test_iter += 1
            print('=====================Epoch {} task1 test acc: {}%, task2 test acc: {}%====================='.format(
                  epoch + 1, round(test_acc1 / test_iter * 100, 2), round(test_acc2 / test_iter * 100, 2)))

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
    use_multilabel = USE_MULTILABEL
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_id = DEVICE_ID
    use_multigpu = USE_MULTIGPU
    device_id_list = DEVICE_ID_LIST
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    max_epoch = MAX_EPOCH
    dataset = DATASET
    input_size = INPUT_SIZE
    model_path = os.getcwd() + "/models/{}_{}.pth".format(model_name, dataset)

    # 得到数据集
    train_loader = get_training_dataset(batch_size, use_multilabel)
    test_loader, classes = get_test_dataset(batch_size, use_multilabel)
    # 构建网络 
    model = load_model(model_name, use_pretrain_model, use_multilabel, classes, input_size)

    # 用GPU运行
    if torch.cuda.device_count() > 1:
        if use_multigpu:
            model = nn.DataParallel(model, device_ids=device_id_list)
            device = torch.device("cuda:{}".format(device_id_list[0]))
        else:
            #os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
            device = torch.device("cuda:{}".format(device_id))
    model = model.to(device)

    # 训练模型
    if use_multilabel:
        model_path = model_path[:-4] + '_multilabel.pth'
        train_multitask_model(model, device, model_path, train_loader, test_loader,
                              batch_size, learning_rate, max_epoch)
    else:
        train_model(model, device, model_path, train_loader, test_loader,
                    batch_size, learning_rate, max_epoch)
    plot_loss()
    plot_acc()

if __name__ == '__main__':
    main()
