import os
import cv2
import json
import yaml
from copy import deepcopy
import requests
import numpy as np

import matplotlib.pyplot as plt

def resize_image(im, keypoints, width, height):
    h, w, c = im.shape
    delta_top = 0
    delta_bottom = 0
    delta_left = 0
    delta_right = 0
    if h > w:
        delta = h - w
        delta_left = delta // 2
        delta_right = delta // 2
        if delta % 2 != 0:
            delta_right += 1
    else:
        delta = w - h
        delta_top = delta // 2
        delta_bottom = delta // 2
        if delta % 2 != 0:
            delta_top += 1
    keypoints_resized = []
    for point in keypoints:
        keypoints_resized.append([point[0], point[1]])
    for point in keypoints_resized:
        point[0] += delta_left
        point[1] += delta_top
    im_resized = cv2.copyMakeBorder(
        im, delta_top, delta_bottom, delta_left, delta_right, cv2.BORDER_CONSTANT, value=0)
    side = im_resized.shape[0]
    scale = 1024.0 / side
    for point_idx, point in enumerate(keypoints_resized):
        keypoints_resized[point_idx] = (point[0] * scale, point[1] * scale)
    im_resized = cv2.resize(im_resized, (width, height))
    keypoints_resized = np.array(keypoints_resized)
    return im_resized, keypoints_resized

def sort_keypoints(keypoints):
    # assumption: left points are on the left
    top_left = [np.inf, np.inf]
    top_right = [0, np.inf]
    bottom_right = [0, 0]
    bottom_left = [np.inf, 0]
    for point in keypoints:
        if point[0] < top_left[0] and point[1] < top_left[1]:
            top_left = point
        elif point[0] > top_right[0] and point[1] < top_right[1]:
            top_right = point
        elif point[0] > bottom_right[0] and point[1] > bottom_right[1]:
            bottom_right = point
        elif point[0] < bottom_left[0] and point[1] > bottom_left[1]:
            bottom_left = point
    return np.array([top_left, top_right, bottom_right, bottom_left])

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
        # sort keypoints: top-left, top-right, bottom-rightm bottom-left
        keypoints = sort_keypoints(keypoints)
        # resize image and keypoints
        im, keypoints = resize_image(im, keypoints, config['width'], config['height'])
        # create bounding box
        bbox = cv2.boundingRect(keypoints.astype(np.int32))
        if vis:
            im_vis = deepcopy(im)
            cv2.rectangle(im_vis, bbox, (255, 0, 0))
            plt.imshow(im_vis)
            plt.show()
        # evaluate homography transform to warp and crop image inside keypoints
        # crop keypoints coordinates
        x_min = np.min(keypoints[:, 0])
        y_min = np.min(keypoints[:, 1])
        keypoints[:, 0] -= x_min
        keypoints[:, 1] -= y_min
        width = bbox[2]
        height = bbox[3]
        keypoints_planar = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)
        h, status = cv2.findHomography(keypoints.astype(np.int32), keypoints_planar)
        im_dst = cv2.warpPerspective(im[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]], h, (width, height))
        # save generated image
        cv2.imwrite(os.path.join(output_path, f'cropped_{image_filename}'), im_dst)
        if vis:
            plt.imshow(im_dst)
            plt.show()

if __name__ == '__main__':
    path_to_config = './config/dataset_generation.yaml'
    with open(path_to_config, 'r') as f:
        config = yaml.safe_load(f)
    parse_labels(config['stage1'])
