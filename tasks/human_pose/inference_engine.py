#!/usr/bin/env python
# coding: utf-8

# First, let's load the JSON file which describes the human pose task.  This is in COCO format, it is the category descriptor pulled from the annotations file.  We modify the COCO category slightly, to add a neck keypoint.  We will use this task description JSON to create a topology tensor, which is an intermediate data structure that describes the part linkages, as well as which channels in the part affinity field each linkage corresponds to.

import os
import json
from glob import glob
from typing import Tuple

import numpy as np
import cv2
import torch
import PIL.Image
import torchvision.transforms as transforms
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from PIL import Image

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


class EnginePredictor:
    def __init__(self, model_path, config=None) -> None:
        with open('human_pose.json', 'r') as f:
            human_pose = json.load(f)

        topology = trt_pose.coco.coco_category_to_topology(human_pose)
        # parse_objects = ParseObjects(topology, cmap_threshold=0.4, link_threshold=0.4, cmap_window=5,
        #                                  line_integral_samples=7, max_num_parts=100, max_num_objects=100)
        self.parse_objects = ParseObjects(topology, cmap_threshold=0.4, link_threshold=0.4, cmap_window=5,
                                        line_integral_samples=7, max_num_parts=100, max_num_objects=100)
        self.draw_objects = DrawObjects(topology)

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        assert os.path.exists(model_path)
        print("Reading engine from file {}".format(model_path))
        with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

    def inference(self, input_file, output_file):
        img = cv2.imread(input_file)
        input_image = preprocess(img)
        input_image = input_image.cpu().numpy()
        image_height, image_width, _ = img.shape

        with self.engine.create_execution_context() as context:
            # Set input shape based on image dimensions for inference
            context.set_binding_shape(self.engine.get_binding_index("input"), (1, 3, image_height, image_width))
            # Allocate host and device buffers
            bindings = []
            output_buffer = []
            output_memory = []
            for binding in self.engine:
                binding_idx = self.engine.get_binding_index(binding)
                size = trt.volume(context.get_binding_shape(binding_idx))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                if self.engine.binding_is_input(binding):
                    # print("input binding informations: ", size, dtype)
                    input_buffer = np.ascontiguousarray(input_image)
                    input_memory = cuda.mem_alloc(input_image.nbytes)
                    bindings.append(int(input_memory))
                else:
                    # print("output binding informations: ", size, dtype)
                    output_buffer.append(cuda.pagelocked_empty(size, dtype))
                    output_memory.append(cuda.mem_alloc(output_buffer[-1].nbytes))
                    bindings.append(int(output_memory[-1]))

            stream = cuda.Stream()
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Transfer prediction output from the GPU.
            cuda.memcpy_dtoh_async(output_buffer[0], output_memory[0], stream)
            cuda.memcpy_dtoh_async(output_buffer[1], output_memory[1], stream)
            # Synchronize the stream
            stream.synchronize()
        cmap = np.reshape(output_buffer[0], (1, 18, 120, 160))
        paf = np.reshape(output_buffer[1], (1, 42, 120, 160))
        cmap, paf = torch.from_numpy(cmap), torch.from_numpy(paf)

        counts, objects, peaks = self.parse_objects(cmap, paf, )
        self.draw_objects(img, counts, objects, peaks)
        cv2.imwrite(out_path, img)
        return cmap


if __name__ == "__main__":
    print("TensorRT version: {}".format(trt.__version__))
    predictor = EnginePredictor("epoch_249_1.engine", 'human_pose.json')
    input_dir = "/home/junhai.yang/pip/images/input/*.jp*g"
    output_dir = "/home/junhai.yang/pip/images/output"
    for input in glob(input_dir):
        out_path = os.path.join(output_dir, os.path.basename(input))
        print("%s --> %s" % (input, out_path))
        predictor.inference(input, out_path)
    del predictor
    