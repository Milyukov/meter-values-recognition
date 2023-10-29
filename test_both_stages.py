import os
import argparse
import time
import cv2
import sys
import tqdm
import requests
import json
import numpy as np
import tensorflow_datasets as tfds

from deploy.utils import parse_analog_detection

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__)) + '/model/'

# Add the parent directory to sys.path
sys.path.append(parent_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--images", help="path to images (1st stage)", type=str)
parser.add_argument("--labels_stage1", help="path to labels.json (1st stage)", type=str)
parser.add_argument("--labels_stage2", help="path to labels.json (2nd stage)", type=str)
parser.add_argument("--dataset", help="path to tesnsorflow dataset", type=str)

ENDPOINT_URL = "http://localhost:8080/recognize"

def extract_info_stage1(labels):
    images_info = []
    filenames_dict = {}
    prefix = 's3://datasets-counters/'
    prefix_length = len(prefix)
    with open(labels) as f:
        images_info = json.load(f)
    for sample in images_info:
        annotation = sample['annotations']
        if len(annotation) == 0:
            continue
        data = sample['data']
        annotation = annotation[0]['result']
        if len(annotation) == 0:
            continue
        element = annotation[0]
        element = element['value']
        class_name = element['polygonlabels'][0]
        filename = data['image'][prefix_length:]
        filenames_dict[filename] = {
            'class_name': class_name
        }
    return filenames_dict


def extract_info_stage2(labels):
    images_info = []
    filenames_dict = {}
    prefix = 's3://stage2/'
    prefix_length = len(prefix)
    with open(labels) as f:
        images_info = json.load(f)
    for sample in images_info:
        annotation = sample['annotations']
        if len(annotation) == 0:
            continue
        data = sample['data']
        annotation = annotation[0]['result']
        bboxes = []
        class_names = []
        for element in annotation:
            width = element['original_width']
            height = element['original_height']
            element = element['value']
            bboxes.append([
                width * element['x'] / 100, 
                height * element['y'] / 100, 
                width * element['width'] / 100, 
                height * element['height'] / 100])
            class_names.append(element['rectanglelabels'][0])
        filename = data['image'][prefix_length:]
        filenames_dict[filename] = {
            'bboxes': bboxes,
            'class_names': class_names
        }
    return filenames_dict

if __name__ == '__main__':
    args = parser.parse_args()
    # get test split from tf dataset
    (test_dataset), dataset_info = tfds.load(
        "meter_values_dataset_stage2", split=["test"], with_info=True, data_dir=args.dataset,
        read_config=tfds.ReadConfig(try_autocache=False)
    )

    test_dataset = test_dataset[0]
    test_filenames = []
    for example in test_dataset:
        filename = example['image/filename'].numpy()
        filename = str(filename).lstrip("b'").rstrip("'")
        test_filenames.append(filename)
    # get path to images for 1st stage
    path_to_images = args.images
    ext = ('.jpg', '.jpeg', '*.png', '*.bmp')
    filenames = [f for f in os.listdir(path_to_images) if f.lower().endswith(ext)]
    # get path to labeling for first stage
    labels_stage1 = args.labels_stage1
    filenames_dict_stage1 = extract_info_stage1(labels_stage1)
    # get path to labeling for 2nd stage
    labels_stage2 = args.labels_stage2
    # map filenames for 2nd stage to meter values text
    filenames_dict_stage2 = extract_info_stage2(labels_stage2)

    # save failed cases
    if not os.path.exists('failures'):
        os.makedirs('failures')

    # loop over images
    timings = []
    number_of_images = 0.0
    number_of_recognized = 0.0
    number_of_recognized_before_fp = 0.0
    for filename in tqdm.tqdm(filenames):
        if not filename in filenames_dict_stage1:
            continue
        if filenames_dict_stage1[filename]['class_name'].lower() != 'analog':
            continue
        # map images for 1st stage to 2nd stage meter values text
        if not 'cropped_' + filename in filenames_dict_stage2:
            continue
        if not 'cropped_' + filename in test_filenames:
            continue
        number_of_images += 1
        st = time.time()
        image = cv2.imread(os.path.join(path_to_images, filename))
        ratio = 500.0 / max(image.shape[:2])
        image = cv2.resize(image, (0, 0), fx = ratio, fy = ratio, interpolation=cv2.INTER_CUBIC)
        data = { 'image': image.tolist() }
        headers = {
            'content-type': "application/json"
        }
        response = requests.post(ENDPOINT_URL, json=data, headers=headers)
        response.raise_for_status()
        response = response.json()
        if not 'text' in response:
            print(response)
        pred_text = response["text"]
        et = time.time()
        timings.append(et - st)
        
        if response['success']:
            info = filenames_dict_stage2['cropped_' + filename]
            bboxes = info['bboxes']

            bboxes_coco19 = []
            for box in bboxes:
                x, y, w, h = box
                bboxes_coco19.append([x, y, x + w, y + h])
            bboxes_coco19 = np.array(bboxes_coco19)

            class_names = info['class_names']
            class_names = list(map(lambda x: x.replace('FLOATP', '14'), class_names))
            gt_text, _, _, _ = parse_analog_detection(bboxes_coco19, np.ones((bboxes_coco19.shape[0],)), class_names)
            if pred_text == gt_text:
                number_of_recognized += 1
            else:
                if ',' in gt_text:
                    gt_integer = gt_text.split(',')[0]
                    if ',' in pred_text:
                        pred_integer = pred_text.split(',')[0]
                    else:
                        pred_integer = pred_text
                    if pred_integer == gt_integer:
                        number_of_recognized_before_fp += 1
                cv2.imwrite(f'./failures/{pred_text}_{filename}', image)
        else:
            cv2.imwrite(f'./failures/{response["counter_class"]}_{filename}', image)
    if number_of_images == 0:
        print('No test files!')
    else:
        print(f'Recognition rate: {number_of_recognized / number_of_images}')
        print(f'Integer recognition rate: {number_of_recognized_before_fp / number_of_images}')
        print(f'Total number of relevant images: {int(number_of_images)}')
        timings = np.array(timings)
        print(f'Percentiles:\n 50% {np.median(timings)}\n90% {np.percentile(timings, 90)}\n99%{np.percentile(timings, 99)}')
