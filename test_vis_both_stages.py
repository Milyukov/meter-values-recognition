import os
import argparse
import time
import cv2
import sys
import requests
import numpy as np

from model.utils_stage2 import visualize_detections 

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__)) + '/model/'

# Add the parent directory to sys.path
sys.path.append(parent_dir)

parser = argparse.ArgumentParser()
parser.add_argument("images", help="path to images", type=str)

ENDPOINT_URL = "http://localhost:8080/recognizeJSON"

if __name__ == '__main__':
    args = parser.parse_args()

    for filename in os.listdir(args.images):
        if not filename.lower().endswith(('.jpg', '.jpeg', '*.png', '*.bmp')):
            continue
        st = time.time()
        image = cv2.imread(os.path.join(args.images, filename))
        ratio = 500.0 / max(image.shape[:2])
        image = cv2.resize(image, (0, 0), fx = ratio, fy = ratio, interpolation=cv2.INTER_CUBIC)
        data = { 'image': image.tolist() }
        headers = {
            'content-type': "application/json"
        }
        response = requests.post(ENDPOINT_URL, json=data, headers=headers)
        response.raise_for_status()
        response = response.json()
        text = response["text"]
        et = time.time()
        print(f'Inference time = {response["time"]}')
        print(f'Total time = {et - st}')
        
        fname = text.replace('.', ',')
        fname = fname + '_' + filename
        if response['success']:
            boxes = np.array(response["bboxes"])
            scores = np.array(response["scores"])
            class_names = np.array(response["class_names"])
            ratio = response["ratio"]
            image_cropped = np.array(response["image_cropped"])
            roi = np.array(response['roi'])
            image_cropped = cv2.rectangle(image_cropped, roi[0], roi[2], (0, 255, 0), 2)
            ax = visualize_detections(
                os.path.join(args.images, 'results', fname),
                image_cropped,
                boxes / ratio,
                class_names,
                scores,
            )
