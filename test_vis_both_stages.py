import os
import argparse
import time

import tensorflow as tf
import cv2

from utils import extract_rectangle_area, parse_analog_detection
import model.utils_stage2 as utils_stage2
from model import model_stage1, data_processing_stage1

import sys

import requests
import json
import numpy as np

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__)) + '/model/'

# Add the parent directory to sys.path
sys.path.append(parent_dir)

parser = argparse.ArgumentParser()
parser.add_argument("images", help="path to images", type=str)

def serialize_image(im):
    predict_request = json.dumps(
        {
            "signature_name": "serving_default", 
            "instances": im.tolist()
        }
    )
    return predict_request

if __name__ == '__main__':
    args = parser.parse_args()
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9 # change this value as needed
    session = tf.compat.v1.Session(config=config)

    for filename in os.listdir(args.images):
        if not filename.lower().endswith(('.jpg', '*.png', '*.bmp')):
            continue
        st = time.time()
        im = cv2.imread(os.path.join(args.images, filename))[:, :, ::-1]
        predict_request = serialize_image(im)
        headers = {"content-type": "application/json"}
        model_url = 'http://localhost:8501/v1/models/stage1:predict'
        json_response = requests.post(model_url, data=predict_request, headers=headers)
        predictions = json.loads(json_response.text)['predictions']
        kept_bboxes = np.array(predictions[0]['bboxes_stage1'])
        kept_scores = np.array(predictions[0]['scores_stage1'])
        image_resized = np.array(predictions[0]['resized_image'])

        if tf.argmax(kept_scores).numpy() == 1 or len(kept_bboxes) == 0:
            continue            

        image_cropped = extract_rectangle_area(image_resized, kept_bboxes[:4], kept_bboxes[4:])
        predict_request = serialize_image(image_cropped)
        headers = {"content-type": "application/json"}
        json_response = requests.post(
            'http://localhost:8502/v1/models/stage2:predict', data=predict_request, headers=headers)
        predictions = json.loads(json_response.text)['predictions']
        kept_bboxes = np.array(predictions[0]['bboxes_stage2'])
        kept_scores = np.array(predictions[0]['scores_stage2'])
        labels = np.array(predictions[0]['labels'])
        ratio = predictions[0]['ratio']

        class_names= [f'{int(x)}' for x in labels]

        text, boxes, scores, class_names = parse_analog_detection(kept_bboxes, kept_scores, class_names)
        fname = text.replace('.', ',')
        fname += '.jpg'

        et = time.time()
        print(f'Inference time = {et - st}')
        ax = utils_stage2.visualize_detections(
            fname,
            image_cropped,
            boxes / ratio,
            class_names,
            scores,
        )
