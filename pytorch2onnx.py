import torch
import torch.onnx
from collections import OrderedDict

from utils import *

def main():
    args = parse_args()
    dataset = "imagenet"
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
    num_classes = 1000
    # 构建网络
    model = MobileNetV3_Small(num_classes)
    
    if use_multilabel:
        model_path = model_path[:-4] + '_multilabel.pth'

    # 加载模型
    if os.path.exists(model_path):
        # 导入模型
        pretrain_model = torch.load(model_path, map_location='cpu')
        state_dict = pretrain_model['state_dict']
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

    # 输入尺寸
    input_size = torch.randn(batch_size, 3, image_size, image_size, requires_grad=True)
    output_onnx = 'onnx/' + model_name + '.onnx'
    torch.onnx.export(model, 
                      input_size, 
                      output_onnx, 
                      verbose=False,    
                      training=False, 
                      input_names=['input'], 
                      output_names=['output'])
    print('Successfully export ONNX Model: {}'.format(output_onnx))
    
if __name__ == '__main__':
    main()
