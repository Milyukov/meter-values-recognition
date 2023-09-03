import argparse
import cv2
from copy import deepcopy
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import yaml


def parse_labels(config):
    input_path = config['path_to_ls_output']
    images_path = config['path_to_images']
    output_path = config['path_to_output']
    vis = config['visualize']
    with open(input_path) as f:
        images_info = json.load(f)
    # for json file
    for image_info in images_info:
        # read image
        image_filename = image_info['file_upload'].split('-')[-1]
        image_path = os.path.join(images_path, image_filename)
        im = cv2.imread(image_path)
        h, w, _ = im.shape
        # read keypoints
        annotations = image_info['annotations']
        keypoints = []
        for field in annotations:
            for result in field['result']:
                if result['type'] == 'keypointlabels':
                    w = result['original_width']
                    h = result['original_height']
                    values = result['value']
                    keypoints.append([
                        values['x'] * w / 100, 
                        values['y'] * h / 100])
        if len(keypoints) == 0:
            continue

        bbox, im_resized, im_dst, im_dst_eq = process_keypoints(
            im, keypoints, config['width'], config['height'])

        # save generated image
        cv2.imwrite(os.path.join(output_path, f'cropped_{image_filename}'), im_dst_eq)

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
