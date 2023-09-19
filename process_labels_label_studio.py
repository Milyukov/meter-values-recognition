import argparse
import cv2
from copy import deepcopy
import json
import os
import matplotlib.pyplot as plt
from utils import *
import yaml

def get_images_info(input_path):
    images_info = []
    with open(input_path) as f:
        images_info = json.load(f)
    return images_info

def generate_examples_stage1(images_info, images_path, width, height):
    # for image
    for image_info in images_info:
        # read image
        image_filename = image_info['file_upload'].split('-')[-1]
        image_path = os.path.join(images_path, image_filename)
        im = cv2.imread(image_path)
        if im is None:
            continue
        # read keypoints
        annotations = image_info['annotations']
        keypoints = []
        labels = []
        for field in annotations:
            for result in field['result']:
                if result['type'] == 'polygonlabels':
                    w = result['original_width']
                    h = result['original_height']
                    values = result['value']
                    labels.append(values['polygonlabels'][0])
                    for point in values['points']:
                        x = point[0]
                        y = point[1]
                        keypoints.append([
                            x * w / 100, 
                            y * h / 100])
        if len(keypoints) == 0:
            continue

        im_resized, bbox, keypoints = process_keypoints(
            im, keypoints, width, height)
        
        yield im_resized, labels, bbox, keypoints, image_filename

def generate_examples_stage2(images_info, images_path, width, height):
    # for image
    for image_info in images_info:
        # read image
        image_filename = image_info['file_upload'].split('-')[-1]
        image_path = os.path.join(images_path, image_filename)
        im = cv2.imread(image_path)
        # read keypoints
        annotations = image_info['annotations']
        bboxes = []
        labels = []
        for field in annotations:
            for result in field['result']:
                if result['type'] == 'rectanglelabels':
                    w = result['original_width']
                    h = result['original_height']
                    values = result['value']
                    labels.append(values['rectanglelabels'][0])
                    x = values['x'] * w / 100
                    y = values['y'] * h / 100
                    bbox_w = values['width'] * w / 100
                    bbox_h = values['height'] * h / 100
                    bboxes.append([x, y, bbox_w, bbox_h])
        if len(bboxes) == 0 or len(labels) == 0:
            continue

        im_resized, bboxes = process_bboxes(
            im, bboxes, width, height)
        
        yield im_resized, labels, bboxes, image_filename

def process_stage1(config):
    index = 0
    # read dataset parameters
    input_path = config['stage1']['path_to_ls_output']
    images_path = config['stage1']['path_to_images']
    width, height = config['stage1']['width'], config['stage1']['height']
    if not os.path.exists(config['stage1']['path_to_output']):
        os.mkdir(config['stage1']['path_to_output'])
    # generate dataset stage 2
    images_info = get_images_info(input_path)
    for im_resized, labels, bbox, keypoints, image_filename in generate_examples_stage1(images_info, images_path, width, height):
        # generate image for stage 2
        im_dst_eq = extract_rectangle_area(im_resized, bbox, keypoints)
        # save generated image
        index += 1
        cv2.imwrite(os.path.join(config['stage1']['path_to_output'], f'cropped_{image_filename}'), im_dst_eq)

        if config['stage1']['visualize']:
            # visualize bbox
            im_vis = deepcopy(im_resized)
            cv2.rectangle(im_vis, bbox, (255, 0, 0))
            # construct figure
            fig, axs = plt.subplots(1, 2, constrained_layout=True)
            fig.suptitle(labels[0])
            axs[0].imshow(im_vis)
            axs[1].imshow(im_dst_eq)
            plt.show()

def process_stage2(config):
    index = 0
    # read dataset parameters
    input_path = config['stage2']['path_to_ls_output']
    images_path = config['stage2']['path_to_images']
    width, height = config['stage2']['width'], config['stage2']['height']
    if not os.path.exists(config['stage2']['path_to_output']):
        os.mkdir(config['stage2']['path_to_output'])
    # generate dataset stage 2
    images_info = get_images_info(input_path)
    for im_resized, labels, bboxes, image_filename in generate_examples_stage2(images_info, images_path, width, height):
        # save generated image
        index += 1
        cv2.imwrite(os.path.join(config['stage2']['path_to_output'], f'resized_{image_filename}'), im_resized)

        if config['stage2']['visualize']:
            # visualize bbox
            im_vis = deepcopy(im_resized)
            for bbox in bboxes:
                cv2.rectangle(
                    im_vis, 
                    (
                        int(bbox[0]), 
                        int(bbox[1])), 
                        (int(bbox[0] + bbox[2]), 
                         int(bbox[1] + bbox[3])), 
                         (255, 0, 0)
                         )
            # construct figure
            fig, axs = plt.subplots(1, 1, constrained_layout=True)
            fig.suptitle(image_filename)
            axs.imshow(im_vis)
            plt.show()

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config", help="Path to *.yaml file with configuration")
    args = argParser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    process_stage1(config)
    process_stage2(config)
