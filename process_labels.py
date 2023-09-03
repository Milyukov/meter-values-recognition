import cv2
import json
import os
import numpy as np

import matplotlib.pyplot as plt

def resize_image(im, keypoints):
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
    im_resized = cv2.resize(im_resized, (1024, 1024))
    keypoints_resized = np.array(keypoints_resized)
    return im_resized, keypoints_resized

if __name__ == '__main__':
    with open('./jsons/project-1-at-2023-08-12-14-30-ee3e4fcd.json') as f:
        images_info = json.load(f)
    images_path = 'C:/Users/cvres/Datasets/Meter/'
    # for json file
    for image_info in images_info:
        # read image
        image_filename = image_info['file_upload']
        image_filename = image_filename.split('-')[-1]
        image_filename = os.path.join(images_path, image_filename)
        im = cv2.imread(image_filename)
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
        # resize image and keypoints
        im, keypoints = resize_image(im, keypoints)
        # create bounding box
        bbox = cv2.boundingRect(keypoints.astype(np.int32))
        im = cv2.rectangle(im, bbox, (255, 0, 0))
        plt.imshow(im)
        plt.show()
        # evaluate homography transform to warp and crop image inside keypoints
        # crop keypoints coordinates
        x_min = np.min(keypoints[:, 0])
        y_min = np.min(keypoints[:, 1])
        keypoints[:, 0] -= x_min
        keypoints[:, 1] -= y_min
        width = bbox[2]
        height = bbox[3]
        keypoints_planar = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.int32)
        h, status = cv2.findHomography(keypoints.astype(np.int32), keypoints_planar)
        im_dst = cv2.warpPerspective(im[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]], h, (width, height))
        #im_dst = cv2.warpPerspective(im, h, (im.shape[1], im.shape[0]))
        # save generated image
        plt.imshow(im_dst)
        plt.show()
