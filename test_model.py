import os
import torch
from PIL import Image
from collections import OrderedDict

from data_preprocess import get_training_dataset, get_test_dataset
from utils import *
from pokemon import Pokemon

# 评估模型
def evaluate_model(model, device, model_path, test_loader, classes, batch_size = 100):
    test_iter_num = len(test_loader)
    test_acc = 0
    test_iter = 0
    ground_truth_dict = dict()
    predict_result_dict = dict()

    for i in range(len(classes)):
        ground_truth_dict[i] = 0
        predict_result_dict[i] = 0

    # 加载模型
    if os.path.exists(model_path):
        # 导入模型
        pretrain_model = torch.load(model_path)
        state_dict = pretrain_model['model_state_dict']
        new_state_dict = OrderedDict()
        is_multigpu_train = False
        for key, value in state_dict.items():
            if key[:6] == "module":
                is_multigpu_train = True
                new_key = key[7:]
                new_state_dict[new_key] = value
            else:
                break

        # 恢复网络的参数
        if is_multigpu_train:
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        print('Successfully Load {} Model'.format(model_path))

    # train时的BN作用和test不一样
    model.eval()
    # 前向计算 不用更新梯度
    with torch.no_grad():
        for test_data in test_loader:
            acc = 0
            test_batch_sample, test_batch_label = test_data
            # 得到测试数据 并传入cuda 默认是cpu to函数转为gpu
            test_batch_sample, test_batch_label = test_batch_sample.to(device), test_batch_label.to(device)
            output = model.forward(test_batch_sample)
            # data得到tensor转成python数组(用于数组)
            _, predict_output = torch.max(output.data, 1)
            correct = (predict_output == test_batch_label).sum().item()
            acc = correct / len(predict_output)
            test_acc += acc
            test_iter += 1
            print('current batch top-1 accuracy is {}%'.format(round(acc * 100.0, 2)))

            for i in range(len(classes)):
                ground_truth_dict[i] += ((test_batch_label == i).sum().item())
            for i in range(predict_output.shape[0]):
                if predict_output[i] == test_batch_label[i]:
                    predict_result_dict[int(predict_output[i].item())] += 1

    print('Test Top-1 Accuracy is: {}%\n'.format(round(test_acc / test_iter * 100.0, 2)))
    for i in range(len(classes)):
        print('{}: {}%({}, {})'.format(classes[i], round(predict_result_dict[i] / ground_truth_dict[i] * 100.0, 2), 
                                       predict_result_dict[i], ground_truth_dict[i]))

# 评估模型
def evaluate_multilabel_model(model, device, model_path, test_loader, classes, batch_size = 100):
    test_iter_num = len(test_loader)
    test_acc1 = 0
    test_acc2 = 0
    test_iter = 0
    ground_truth_dict1 = dict()
    ground_truth_dict2 = dict()
    predict_result_dict1 = dict()
    predict_result_dict2 = dict()

    for i in range(len(classes[0])):
        ground_truth_dict1[i] = 0
        predict_result_dict1[i] = 0
    for i in range(len(classes[1])):
        ground_truth_dict2[i] = 0
        predict_result_dict2[i] = 0

    # 加载模型
    if os.path.exists(model_path):
        # 导入模型
        pretrain_model = torch.load(model_path)
        state_dict = pretrain_model['model_state_dict']
        new_state_dict = OrderedDict()
        is_multigpu_train = False
        for key, value in state_dict.items():
            if key[:6] == "module":
                is_multigpu_train = True
                new_key = key[7:]
                new_state_dict[new_key] = value
            else:
                break

        # 恢复网络的参数
        if is_multigpu_train:
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        print('Successfully Load {} Model'.format(model_path))

    # train时的BN作用和test不一样
    model.eval()
    # 前向计算 不用更新梯度
    with torch.no_grad():
        for test_data in test_loader:
            acc = 0
            test_batch_sample, test_batch_label = test_data
            # 得到测试数据 并传入cuda 默认是cpu to函数转为gpu
            test_batch_sample, test_batch_label = test_batch_sample.to(device), test_batch_label.to(device)
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
            print('current batch task1 top-1 accuracy is {}%, task2 top-1 accuracy is {}%'.format(
                   round(acc1 * 100.0, 2), round(acc2 * 100.0, 2)))

            for i in range(len(classes[0])):
                ground_truth_dict1[i] += ((label1 == i).sum().item())
            for i in range(len(classes[1])):
                ground_truth_dict2[i] += ((label2 == i).sum().item())
            for i in range(predict_output1.shape[0]):
                if predict_output1[i] == label1[i]:
                    predict_result_dict1[int(predict_output1[i].item())] += 1
            for i in range(predict_output2.shape[0]):
                if predict_output2[i] == label2[i]:
                    predict_result_dict2[int(predict_output2[i].item())] += 1

    print('Test Task1 Top-1 Accuracy is: {}%, Task2 Top-1 Accuracy is: {}%\n'.format(
           round(test_acc1 / test_iter * 100.0, 2), round(test_acc2 / test_iter * 100.0, 2)))
    for i in range(len(classes[0])):
        print('{}: {}%({}, {})'.format(classes[0][i], round(predict_result_dict1[i] / ground_truth_dict1[i] * 100.0, 2),
                                       predict_result_dict1[i], ground_truth_dict1[i]))
    for i in range(len(classes[1])):
        try:
            print('{}: {}%({}, {})'.format(classes[1][i], round(predict_result_dict2[i] / ground_truth_dict2[i] * 100.0, 2),
                                       predict_result_dict2[i], ground_truth_dict2[i]))
        except Exception as e:
            continue

# 测试一张图片的推理结果
def test_model_inference(model, device, model_path, resize_h, resize_w, classes, use_multilabel):
    pokemon = Pokemon(POKEMON_DIR, resize_h, resize_w, None, use_multilabel)
    images_list, _ = pokemon.load_csv(os.path.join(POKEMON_DIR, 'pokemon_multilabel.csv'))
    #image_path = os.path.join(POKEMON_DIR, '4杰尼龟', '00000020.jpg')
    #images_list = [image_path]

    # 加载模型
    if os.path.exists(model_path):
        # 导入模型
        pretrain_model = torch.load(model_path)
        state_dict = pretrain_model['model_state_dict']
        new_state_dict = OrderedDict()
        is_multigpu_train = False
        for key, value in state_dict.items():
            if key[:6] == "module":
                is_multigpu_train = True
                new_key = key[7:]
                new_state_dict[new_key] = value
            else:
                break

        # 恢复网络的参数
        if is_multigpu_train:
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        print('Successfully Load {} Model'.format(model_path))

    # train时的BN作用和test不一样
    model.eval()
    # 前向计算 不用更新梯度
    with torch.no_grad():
        for image_path in images_list:
            image = Image.open(image_path).convert('RGB') #png是RGBA所以要转换
            # 对图片预处理
            image = pokemon.transform(image)
            # c h w在第一维上添加一个维度变成n c h w
            image = image.unsqueeze(0)
        
            image = image.to(device)
            output = model.forward(image)
            if use_multilabel:
                output1, output2 = output
                # data得到tensor转成python数组(用于数组)
                _, predict_output1 = torch.max(output1.data, dim=1)
                _, predict_output2 = torch.max(output2.data, dim=1)
                print(image_path, ':', classes[0][int(predict_output1)], classes[1][int(predict_output2)])
            else:
                _, predict_output = torch.max(output.data, dim=1)
                print(image_path, ':', classes[int(predict_output)])

def main():
    args = parse_args()
    dataset = args.dataset
    model_name = args.model_name
    use_pretrain_model = args.use_pretrain_model
    use_multilabel = args.use_multilabel
    test_mode = args.test_mode

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_id = args.device_id

    image_size = args.image_size
    resize_h = args.resize_h
    resize_w = args.resize_w
    batch_size = args.batch_size
    model_path = os.getcwd() + "/models/{}_{}.pth".format(model_name, dataset)

    if dataset == "cifar10":
        image_size = 32
    input_size = int(512 * (4 + (image_size / 32) - 1) * (4 + (image_size / 32) - 1))

    # 得到数据集
    test_loader, classes = get_test_dataset(dataset, batch_size, resize_h, resize_w, use_multilabel)
    # 构建网络
    model = create_model(model_name, use_pretrain_model, use_multilabel, classes, input_size)
    
    # 用GPU运行
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:{}".format(device_id))
    model = model.to(device)

    if use_multilabel:
        model_path = model_path[:-4] + '_multilabel.pth'

    # 评估模型还是测试模型推理结果(单张图)
    if test_mode == "eval":
        if use_multilabel:
            evaluate_multilabel_model(model, device, model_path, test_loader, classes, batch_size)
        else:
            evaluate_model(model, device, model_path, test_loader, classes, batch_size)
    elif test_mode == "test":
        test_model_inference(model, device, model_path, resize_h, resize_w, classes, use_multilabel)
    
if __name__ == '__main__':
    main()
