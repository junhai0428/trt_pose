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

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
parse_objects = ParseObjects(topology, cmap_threshold=0.1, link_threshold=0, cmap_window=5,
                                 line_integral_samples=7, max_num_parts=100, max_num_objects=100)
draw_objects = DrawObjects(topology)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])
model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).to(device).eval()
MODEL_WEIGHTS = 'epoch_249.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))

session = onnxruntime.InferenceSession('epoch_249.onnx', None)

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])

def preprocess(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    image = image.numpy()
    return image.reshape(1, 3, 320, 320)


def execute(image_path, output_path):
    with torch.no_grad():
        out_path = os.path.join(output_path, os.path.basename(image_path))
        image = cv2.imread(image_path)
        data = preprocess(image)

        cmap, paf = session.run([], {'input': data})
        cmap, paf = cmap.reshape(18, 80, 80), paf.reshape(42, 80, 80)
        print("after onn run: ", type(cmap), type(paf))
        print("after onn run: ", cmap.shape, paf.shape)

        # counts, objects, peaks = parse_objects(cmap, paf, )
        # draw_objects(image, counts, objects, peaks)
        # cv2.imwrite(out_path, image)
        
        for i, feature in enumerate(cmap):
            print(type(feature))
            print(feature.shape)
            print(feature.max(), feature.min())
            feature = 255 * (feature - feature.min()) / (feature.max() - feature.min())
            cv2.imwrite(f"/home/junhai_yang/pip/images/onnx_feature_{i+1}.jpg", feature.astype(np.uint8))
            # break


input_path = "/home/junhai_yang/pip/images/input/*.jpg"
output_path = "/home/junhai_yang/pip/images/output"
for file in glob(input_path):
    print(file)
    execute(file, output_path)
