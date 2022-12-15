import json
import torch
import trt_pose.coco
import trt_pose.models


def torch2onnx(model_torch, model_onnx, input_size):
    print("start")

    input_name = ['input']
    output_name = ['output1', 'output2']
    '''input为输入模型图片的大小'''
    input = torch.zeros((1, 3, input_size, input_size)).cuda()

    model = trt_pose.models.densenet121_baseline_att(18, 42).cuda().eval()

    print("read model weights")
    model.load_state_dict(torch.load(model_torch))
    print(model)

    # print("transfer")
    # torch.onnx.export(model, input, model_onnx, input_names=input_name, output_names=output_name, verbose=True)


if __name__ == "__main__":
    model_torch = "./epoch_249.pth"
    model_onnx = "./epoch_249.onnx"
    torch2onnx(model_torch, model_onnx, 320)
