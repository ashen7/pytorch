import torch
import torch.onnx

def main():
    model_name = MODEL_NAME
    use_pretrain_model = USE_PRETRAIN_MODEL
    use_multilabel = USE_MULTILABEL
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_id = DEVICE_ID
    batch_size = BATCH_SIZE
    dataset = DATASET
    input_size = INPUT_SIZE
    test_mode = "test"
    model_path = os.getcwd() + "/models/{}_{}.pth".format(model_name, dataset)

    # 得到数据集
    test_loader, classes = get_test_dataset(batch_size, use_multilabel)
    # 构建网络
    model = load_model(model_name, use_pretrain_model, use_multilabel, classes, input_size)
    
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
        test_model_inference(model, device, model_path, classes, use_multilabel)
    
if __name__ == '__main__':
    main()
