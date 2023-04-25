#!/usr/bin/env python
# coding: utf-8

# First, let's load the JSON file which describes the human pose task.  This is in COCO format, it is the category descriptor pulled from the annotations file.  We modify the COCO category slightly, to add a neck keypoint.  We will use this task description JSON to create a topology tensor, which is an intermediate data structure that describes the part linkages, as well as which channels in the part affinity field each linkage corresponds to.

import os
import json
from glob import glob

import numpy as np
import onnxruntime   # to inference ONNX models, we use the ONNX Runtime
import onnx
import cv2
import torch
import PIL.Image
import torchvision.transforms as transforms

import trt_pose.coco
import trt_pose.models
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda')


def preprocess(image):
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


class OnnxPredictor:
    def __init__(self, model_path, config=None) -> None:
        with open('human_pose.json', 'r') as f:
            human_pose = json.load(f)

        topology = trt_pose.coco.coco_category_to_topology(human_pose)
        # parse_objects = ParseObjects(topology, cmap_threshold=0.4, link_threshold=0.4, cmap_window=5,
        #                                  line_integral_samples=7, max_num_parts=100, max_num_objects=100)
        self.parse_objects = ParseObjects(topology, cmap_threshold=0.6, link_threshold=0.4, cmap_window=5,
                                        line_integral_samples=7, max_num_parts=100, max_num_objects=100)
        self.draw_objects = DrawObjects(topology)

        self.session = onnxruntime.InferenceSession(model_path, None)


    def inference(self, input_path, output_path):
        with torch.no_grad():
            image = cv2.imread(input_path, flags=cv2.IMREAD_COLOR)
            data = preprocess(image)
            data = data.cpu().numpy()
 
            cmap, paf = self.session.run([], {'input': data})
            cmap, paf = torch.from_numpy(cmap), torch.from_numpy(paf)
            print(type(cmap), type(paf))
            print(cmap.shape, paf.shape)

            counts, objects, peaks = self.parse_objects(cmap, paf, )
            self.draw_objects(image, counts, objects, peaks)
            cv2.imwrite(output_path, image)
        return cmap.numpy()

if __name__ == "__main__":
    predictor = OnnxPredictor("epoch_249.onnx", 'human_pose.json')
    input_dir = "/home/junhai.yang/pip/images/input/*.jp*g"
    output_dir = "/home/junhai.yang/pip/images/output"
    for input in glob(input_dir):
        out_path = os.path.join(output_dir, os.path.basename(input))
        print("%s --> %s" % (input, out_path))
        predictor.inference(input, out_path)
    del predictor
