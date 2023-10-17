import os
import argparse
import numpy as np
import time

import tensorflow as tf
import cv2

from utils import extract_rectangle_area, parse_analog_detection

import model.data_processing_stage1 as data_processing_stage1
import model.model_stage1 as model_stage1

import model.data_processing_stage2 as data_processing_stage2
import model.model_stage2 as model_stage2
import model.utils_stage2 as utils_stage2

import sys

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__)) + '/model/'

# Add the parent directory to sys.path
sys.path.append(parent_dir)

parser = argparse.ArgumentParser()
parser.add_argument("model_stage1", help="absolute path to model files", type=str)
parser.add_argument("model_stage2", help="absolute path to model files", type=str)
parser.add_argument("images", help="path to images", type=str)

def get_iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = tf.maximum(ground_truth[0], pred[0])
    iy1 = tf.maximum(ground_truth[1], pred[1])
    ix2 = tf.minimum(ground_truth[2], pred[2])
    iy2 = tf.minimum(ground_truth[3], pred[3])
    
    # Intersection height and width.
    i_height = tf.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = tf.maximum(ix2 - ix1 + 1, np.array(0.))
    
    area_of_intersection = i_height * i_width
    
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
    
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
    
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    
    iou = area_of_intersection / area_of_union
    
    return iou

def prepare_image_stage1(image):
    image, _, ratio = data_processing_stage1.resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

def prepare_image_stage2(image):
    image, _, ratio = data_processing_stage2.resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

def apply_nms(bboxes, confidences):
    max_detections = 5
    # select only rectangles above a confidence threshold
    max_confidences = tf.reduce_max(confidences , axis=1, keepdims=True)
    valid_detections = max_confidences >= 0.0
    confidences = confidences[valid_detections[:, 0]]
    bboxes = bboxes[valid_detections[:, 0]]
    # sort the thresholded rectangles in descending order
    sorted_indices = tf.argsort(max_confidences, direction='DESCENDING', axis=0)
    sorted_scores = tf.gather(confidences, sorted_indices[:, 0])
    sorted_bboxes = tf.gather(bboxes, sorted_indices[:, 0])
    # create an empty set of kept rectangle
    kept_bboxes = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    kept_scores = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    # loop over the sorted thresholded rectangles
    max_detections_val = tf.minimum(max_detections, len(sorted_indices))
    kept_size = 0
    for bbox_index in range(max_detections_val):
        bbox = sorted_bboxes[bbox_index]
        # loop over the set of kept rectangles:
        all_iou_lower = True
        for kept_bbox_index in range(kept_bboxes.size()):
            kept_bbox = kept_bboxes.read(kept_bbox_index)
            # compute IOU between the rectangles
            iou = get_iou(bbox, kept_bbox)
            # if IOU is above IOU threshold break loop
            if iou > 0.7:
                all_iou_lower = False
                break
        # if all IOU are below the IOU threshold add to kept
        if all_iou_lower:
            kept_bboxes.write(kept_size, bbox)
            kept_scores.write(kept_size, sorted_scores[bbox_index])
            kept_size += 1
    return kept_bboxes, kept_scores

if __name__ == '__main__':
    args = parser.parse_args()
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # change this value as needed
    session = tf.compat.v1.Session(config=config)

    label_encoder_stage1 = data_processing_stage1.LabelEncoder()
    label_encoder_stage2 = data_processing_stage2.LabelEncoder()
    num_classes_stage1 = 4
    num_classes_stage1 = 17
    batch_size = 1

    checkpoint_path_stage1 = args.model_stage1
    checkpoint_path_stage2 = args.model_stage2
    if os.path.exists(checkpoint_path_stage1) and os.path.exists(checkpoint_path_stage2):
        stage1 = tf.keras.saving.load_model(checkpoint_path_stage1)
        stage2 = tf.keras.saving.load_model(checkpoint_path_stage2)

        image = tf.keras.Input(shape=[None, None, 3], name="image")
        predictions = stage1(image, training=False)
        detections = model_stage1.DecodePredictions(confidence_threshold=0.5)(image, predictions)
        inference_model_stage1 = tf.keras.Model(inputs=image, outputs=detections)

        image = tf.keras.Input(shape=[None, None, 3], name="image")
        predictions = stage2(image, training=False)
        detections = model_stage2.DecodePredictions(confidence_threshold=0.5)(image, predictions)
        inference_model_stage2 = tf.keras.Model(inputs=image, outputs=detections)

        max_detections = 5

        for filename in os.listdir(args.images):
            if not filename.lower().endswith(('.jpg', '*.png', '*.bmp')):
                continue
            st = time.time()
            image = cv2.imread(os.path.join(args.images, filename))
            image_resized, _, _ = data_processing_stage1.resize_and_pad_image(image, jitter=None)
            image_resized = image_resized.numpy()
            input_image = tf.cast(image, dtype=tf.float32)
            input_image, ratio = prepare_image_stage1(input_image)
            print(input_image.shape)
            boxes, cls_predictions = inference_model_stage1.predict(input_image)

            kept_bboxes, kept_scores = apply_nms(boxes[0], cls_predictions[0])
            kept_bboxes = kept_bboxes.read(0)
            kept_scores = kept_scores.read(0)

            if tf.argmax(kept_scores).numpy() % 2 == 1:
                continue            

            image_cropped = extract_rectangle_area(image_resized, kept_bboxes[:4], kept_bboxes[4:])
            input_image = tf.cast(image_cropped, dtype=tf.float32)
            input_image, ratio = prepare_image_stage2(input_image)
            print(input_image.shape)

            detections = inference_model_stage2.predict(input_image)
            text, boxes, scores, class_names = parse_analog_detection(detections)
            fname = text.replace('.', ',')
            fname += '.jpg'

            et = time.time()
            print(f'Inference time = {et - st}')
            num_detections = detections.valid_detections[0]
            class_names = [f'{int(x)}' for x in detections.nmsed_classes[0][:num_detections]]
            ax = utils_stage2.visualize_detections(
                fname,
                image_cropped,
                boxes / ratio,
                class_names,
                scores,
            )
