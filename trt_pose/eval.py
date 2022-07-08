import argparse
import torchvision
import os
import torch.optim
import apex.amp as amp
import json
import pprint
import pycocotools

import PIL.Image
import trt_pose
from trt_pose.coco import CocoDataset
from trt_pose.models import MODELS
from trt_pose.parse_objects import ParseObjects

OPTIMIZERS = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam
}

device = torch.device('cuda')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    
    print('Loading config %s' % args.config)
    with open(args.config, 'r') as f:
        config = json.load(f)
        pprint.pprint(config)

    model = MODELS[config['model']['name']](**config['model']['kwargs']).to(device)
    optimizer = OPTIMIZERS[config['optimizer']['name']](model.parameters(), **config['optimizer']['kwargs'])
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # ******************************************************************************
    model = torch.nn.DataParallel(model, device_ids=[2])
    model.load_state_dict(torch.load('/media/nio/storage/pip/trt_pose/tasks/human_pose/experiments'
                                     '/mobilenet_baseline_att_320x320_A.json.checkpoints/epoch_60.pth'))
    # ******************************************************************************

    model = model.eval()

    IMAGE_SHAPE = (320, 320)
    images_dir = '/media/nio/storage/coco/val2017'
    annotation_file = './annotations/person_keypoints_val2017_modified.json'

    cocoGtTmp = pycocotools.coco.COCO('./annotations/person_keypoints_val2017_modified.json')
    topology = trt_pose.coco.coco_category_to_topology(cocoGtTmp.cats[1])
    cocoGt = pycocotools.coco.COCO('./annotations/person_keypoints_val2017.json')

    catIds = cocoGt.getCatIds('person')
    imgIds = cocoGt.getImgIds(catIds=catIds)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    parse_objects = ParseObjects(topology, cmap_threshold=0.05, link_threshold=0.1, cmap_window=11,
                                 line_integral_samples=7, max_num_parts=100, max_num_objects=100)

    results = []

    for n, imgId in enumerate(imgIds):

        # read image
        img = cocoGt.imgs[imgId]
        img_path = os.path.join(images_dir, img['file_name'])

        image = PIL.Image.open(img_path).convert('RGB').resize(IMAGE_SHAPE)
        data = transform(image).cuda()[None, ...]

        cmap, paf = model(data)
        cmap, paf = cmap.cpu(), paf.cpu()

        object_counts, objects, peaks = parse_objects(cmap, paf)
        object_counts, objects, peaks = int(object_counts[0]), objects[0], peaks[0]

        for i in range(object_counts):
            _object = objects[i]
            score = 0.0
            kps = [0] * (17 * 3)
            x_mean = 0
            y_mean = 0
            cnt = 0
            for j in range(17):
                k = _object[j]
                if k >= 0:
                    peak = peaks[j][k]
                    x = round(float(img['width'] * peak[1]))
                    y = round(float(img['height'] * peak[0]))
                    score += 1.0
                    kps[j * 3 + 0] = x
                    kps[j * 3 + 1] = y
                    kps[j * 3 + 2] = 2
                    x_mean += x
                    y_mean += y
                    cnt += 1

            ann = {
                'image_id': imgId,
                'category_id': 1,
                'keypoints': kps,
                'score': score / 17.0
            }
            results.append(ann)
        if n % 100 == 0:
            print('%d / %d' % (n, len(imgIds)))

    with open('results.json', 'w') as f:
        json.dump(results, f)

    cocoDt = cocoGt.loadRes('results.json')

    cocoEval = pycocotools.cocoeval.COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.params.imgIds = imgIds
    cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
