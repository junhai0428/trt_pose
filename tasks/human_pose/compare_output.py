import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from inference import PytorchPredictor
from inference_onnx import OnnxPredictor

def compare(data1, data2):
    e = (data1 - data2).flatten()
    print("#########################################")
    print("min: %f, mean: %f, max: %f" % (np.min(e), np.mean(e), np.max(e)))
    print("std: %f" % np.std(e))
    print("#########################################")

    plt.figure(1)
    plt.hist(e, bins=20, log=True, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("x")
    # 显示纵轴标签
    plt.ylabel("frequency")
    plt.savefig('squares_plot.png', bbox_inches='tight')


if __name__ == "__main__":
    # data1 = np.load("cmap.npy")
    # data2 = np.load("output_1.npy")

    # compare(data1, data2)
    predictor1 = PytorchPredictor("epoch_249.pth", 'human_pose.json')
    predictor2 = OnnxPredictor("epoch_249.onnx", 'human_pose.json')
    input_dir = "/home/junhai.yang/pip/images/input/*.jp*g"
    output_dir = "/home/junhai.yang/pip/images/output"
    for input in glob(input_dir):
        out_path = os.path.join(output_dir, os.path.basename(input))
        print("%s --> %s" % (input, out_path))
        output1 = predictor1.inference(input, out_path)
        output2 = predictor2.inference(input, out_path)
        np.save("output1.txt", output1)
        np.save("output2.txt", output2)
        compare(output1, output2)
        break
    del predictor1, predictor2
