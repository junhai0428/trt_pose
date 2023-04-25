#!/usr/bin/env python
# coding: utf-8

# First, let's load the JSON file which describes the human pose task.  This is in COCO format, it is the category descriptor pulled from the annotations file.  We modify the COCO category slightly, to add a neck keypoint.  We will use this task description JSON to create a topology tensor, which is an intermediate data structure that describes the part linkages, as well as which channels in the part affinity field each linkage corresponds to.

import os
import json
import time
from glob import glob

import numpy as np
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


class PytorchPredictor:
    def __init__(self, model_path, config=None) -> None:
        with open(config, 'r') as f:
            human_pose = json.load(f)

        topology = trt_pose.coco.coco_category_to_topology(human_pose)
        # parse_objects = ParseObjects(topology, cmap_threshold=0.4, link_threshold=0.4, cmap_window=5,
        #                                  line_integral_samples=7, max_num_parts=100, max_num_objects=100)
        self.parse_objects = ParseObjects(topology, cmap_threshold=0.4, link_threshold=0.4, cmap_window=5,
                                        line_integral_samples=7, max_num_parts=100, max_num_objects=100)
        self.draw_objects = DrawObjects(topology)

        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])

        self.model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).to(device).eval()
        self.model.load_state_dict(torch.load(model_path))

    def inference(self, input_path, output_path):

        with torch.no_grad():
            image = cv2.imread(input_path)
            start = time.time()
            data = preprocess(image)
            print("preprocess time: ", time.time() - start)

            cmap, paf = self.model(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            np.save("cmap", cmap)
            np.save("paf", paf)
            print(type(cmap), type(paf))
            print(cmap.shape, paf.shape)


            counts, objects, peaks = self.parse_objects(cmap, paf, )
            self.draw_objects(image, counts, objects, peaks)
            cv2.imwrite(output_path, image)
            
            # cmap, paf = cmap.detach().cpu().squeeze().numpy(), paf.detach().cpu().squeeze().numpy()
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>", cmap.shape, paf.shape)
            # for i, feature in enumerate(cmap):
            #     print(type(feature))
            #     print(feature.shape)
            #     print(feature.max(), feature.min())
            #     feature = 255 * (feature - feature.min()) / (feature.max() - feature.min())
            #     cv2.imwrite(f"/home/junhai_yang/pip/images/feature_{i+1}.jpg", feature.astype(np.uint8))
                # break
        return cmap.numpy()

if __name__ == "__main__":
    predictor = PytorchPredictor("epoch_249.pth", 'human_pose.json')
    input_dir = "/home/junhai.yang/pip/images/input/*.jp*g"
    output_dir = "/home/junhai.yang/pip/images/output"
    for input in glob(input_dir):
        out_path = os.path.join(output_dir, os.path.basename(input))
        print("%s --> %s" % (input, out_path))
        predictor.inference(input, out_path)
    del predictor
