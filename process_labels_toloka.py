import argparse
import cv2
from copy import deepcopy
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import requests
from utils import *
import yaml


def parse_labels(config):
    input_path = config['path_to_toloka_output']
    output_path = config['path_to_output']
    vis = config['visualize']
    with open(input_path) as f:
        images_info = json.load(f)
    # for json file
    for image_info in images_info:
        annotations = image_info['output_values']
        if annotations['result_types'] in ['illegible', '_404']:
            continue
        # read image
        image_url = image_info['input_values']['image']
        image_filename = image_url.split('/')[-1]
        response = requests.get(image_url)
        arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        im = cv2.imdecode(arr, -1) # 'Load it as it is'
        h, w, _ = im.shape
        # read keypoints
        keypoints = []
        for point in annotations['result_points']:
            if point['shape'] == 'point':
                keypoints.append([point['left'] * w, point['top'] * h])
        bbox, im_resized, im_dst, im_dst_eq = process_keypoints(
            im, keypoints, config['width'], config['height'])
        # save generated image
        cv2.imwrite(os.path.join(output_path, f'cropped_{image_filename}'), im_dst)
        if vis:
            im_vis = deepcopy(im_resized)
            cv2.rectangle(im_vis, bbox, (255, 0, 0))
            plt.imshow(im_vis)
            plt.show()
            plt.subplot(121)
            plt.imshow(im_dst)
            plt.subplot(122)
            plt.imshow(im_dst_eq)
            plt.show()

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config", help="Path to *.yaml file with configuration")
    args = argParser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    parse_labels(config['stage1'])
