import os
import argparse
import time
import cv2
import sys
import requests


# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__)) + '/model/'

# Add the parent directory to sys.path
sys.path.append(parent_dir)

parser = argparse.ArgumentParser()
parser.add_argument("images", help="path to images", type=str)

ENDPOINT_URL = "http://172.17.0.2:8080/recognize"

if __name__ == '__main__':
    args = parser.parse_args()

    for filename in os.listdir(args.images):
        if not filename.lower().endswith(('.jpg', '*.png', '*.bmp')):
            continue
        st = time.time()
        image = cv2.imread(os.path.join(args.images, filename))
        data = { 'image': image.tolist() }
        headers = {
            'content-type': "application/json"
        }
        response = requests.post(ENDPOINT_URL, json=data, headers=headers)
        response.raise_for_status()
        response = response.json()
        text = response["text"]
        print(f'{filename}: {response}')
        et = time.time()
        print(f'Inference time = {et - st}')
        
        # fname = text.replace('.', ',')
        # fname += '.jpg'
        # ax = utils_stage2.visualize_detections(
        #     fname,
        #     image_cropped,
        #     boxes / ratio,
        #     class_names,
        #     scores,
        # )
