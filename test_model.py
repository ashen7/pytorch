import os
import torch
from PIL import Image
from collections import OrderedDict

from data_preprocess import get_training_dataset, get_test_dataset
from utils import *
from pokemon import Pokemon

# 评估模型
def evaluate_model(model, device, model_path, test_loader, classes, batch_size = 100):
    test_batch_num = len(test_loader)
    test_acc = 0
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
            acc = (predict_output == test_batch_label).sum().item()
            print('current batch top-1 accuracy is {}%'.format(acc))
            test_acc += acc

            for i in range(len(classes)):
                ground_truth_dict[i] += ((test_batch_label == i).sum().item())
            for i in range(predict_output.shape[0]):
                if predict_output[i] == test_batch_label[i]:
                    predict_result_dict[int(predict_output[i].item())] += 1

    print('Test Top-1 Accuracy is: {}%\n'.format(test_acc / test_batch_num))
    for i in range(len(classes)):
        print('{}: {}%'.format(classes[i], round(predict_result_dict[i] / ground_truth_dict[i] * 100.0, 2)))


# 测试一张图片的推理结果
def test_model_inference(model, device, model_path):
    image_path = os.path.join(POKEMON_DIR, '4杰尼龟', '00000020.jpg')

    # 加载模型
    if os.path.exists(model_path):
        # 导入模型
        pretrain_model = torch.load(model_path)
        # 恢复网络的参数
        model.load_state_dict(pretrain_model['model_state_dict'])
        print('Successfully Load {} Model'.format(model_path))

    # train时的BN作用和test不一样
    model.eval()
    # 前向计算 不用更新梯度
    with torch.no_grad():
        image = Image.open(image_path)
        # 对图片预处理 
        pokemon = Pokemon(POKEMON_DIR, RESIZE_HEIGHT, RESIZE_WIDTH)
        image = pokemon.transform(image)
        # c h w在第一维上添加一个维度变成n c h w
        image = image.unsqueeze(0)
        
        image = image.to(device)
        output = model.forward(image)
        # data得到tensor转成python数组(用于数组)
        _, predict_output = torch.max(output.data, 1)
        print(image_path, ':', classes[int(predict_output)])


def main():
    model_name = MODEL_NAME
    use_pretrain_model = USE_PRETRAIN_MODEL
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_id = 0
    batch_size = BATCH_SIZE
    dataset = DATASET
    model_path = os.getcwd() + "/models/{}_{}.pth".format(model_name, dataset)

    # 得到数据集
    test_loader, classes = get_test_dataset(batch_size)
    # 构建网络
    model = load_model(model_name, use_pretrain_model, device, device_id) 
    print(model)

    # 评估模型还是测试模型推理结果(单张图)
    if TEST_MODE == "eval":
        evaluate_model(model, device, model_path, test_loader, classes, batch_size)
    elif TEST_MODE == "test":
        test_model_inference(model, device, model_path)
    
if __name__ == '__main__':
    main()
