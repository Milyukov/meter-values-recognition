import os
import argparse
import cv2
import sys
import requests
import numpy as np

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__)) + '/model/'

# Add the parent directory to sys.path
sys.path.append(parent_dir)

parser = argparse.ArgumentParser()
parser.add_argument("images", help="path to images", type=str)
parser.add_argument("output", help="path to cropped images", type=str)

ENDPOINT_URL = "http://localhost:8000/recognize"

if __name__ == '__main__':
    args = parser.parse_args()

    for filename in os.listdir(args.images):
        if not filename.lower().endswith(('.jpg', '*.png', '*.bmp')):
            continue
        image = cv2.imread(os.path.join(args.images, filename))
        ratio = 1333.0 / max(image.shape[:2])
        image = cv2.resize(image, (0, 0), fx = ratio, fy = ratio, interpolation=cv2.INTER_CUBIC)
        data = { 'image': image.tolist() }
        headers = {
            'content-type': "application/json"
        }
        response = requests.post(ENDPOINT_URL, json=data, headers=headers)
        response.raise_for_status()
        response = response.json()
        if response['success']:
            boxes = np.array(response["bboxes"])
            scores = np.array(response["scores"])
            class_names = np.array(response["class_names"])
            ratio = response["ratio"]
            image_cropped = np.array(response["image_cropped"])
            cv2.imwrite(os.path.join(args.output, 'cropped_' + filename))
