# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors,
# and stored with the default serving key
import tempfile
import tensorflow as tf
import os
import sys

import numpy as np
import cv2

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__)) + '/../'

# Add the parent directory to sys.path
sys.path.append(parent_dir)
sys.path.append(parent_dir + 'model')

# Import the module from the parent directory
from model import model_stage1, data_processing_stage1
from model import model_stage2

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

def extract_rectangle_area(im_resized, bbox, keypoints):
    # evaluate homography transform to warp and crop image inside keypoints
    # crop keypoints coordinates
    x_min = np.min(keypoints[::2])
    y_min = np.min(keypoints[1::2])
    keypoints = np.array([
        [keypoints[0], keypoints[1]], 
        [keypoints[2], keypoints[3]], 
        [keypoints[4], keypoints[5]], 
        [keypoints[6], keypoints[7]]])
    keypoints[:, 0] -= x_min
    keypoints[:, 1] -= y_min
    width = int(np.sqrt((keypoints[0][0] - keypoints[1][0]) ** 2 + (keypoints[0][1] - keypoints[1][1]) ** 2))
    height = int(np.sqrt((keypoints[0][0] - keypoints[3][0]) ** 2 + (keypoints[0][1] - keypoints[3][1]) ** 2))
    keypoints_planar = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)

    h, status = cv2.findHomography(keypoints.astype(np.int32), keypoints_planar)
    # for extended area
    h_inv, status = cv2.findHomography(keypoints_planar, keypoints.astype(np.int32))

    # extend area
    keypoints_planar_extended = keypoints_planar + np.array(
        [[-height//2, -2 * height//3], 
        [height//2, -2 * height//3], 
        [height//2, 2 * height//3], 
        [-height//2, 2 * height//3]])
    # use inverse homography to choose new points on the original image
    keypoints_planar_extended = keypoints_planar_extended.reshape(-1,1,2).astype(np.float32)
    keypoints_extended = cv2.perspectiveTransform(keypoints_planar_extended, h_inv)
    keypoints_extended = keypoints_extended.reshape(-1, 2)
    keypoints_extended[:, 0] += x_min
    keypoints_extended[:, 1] += y_min

    # get bbox
    bbox = cv2.boundingRect(keypoints_extended.astype(np.int32))
    width = int(np.sqrt((keypoints_extended[0][0] - keypoints_extended[1][0]) ** 2 + (keypoints_extended[0][1] - keypoints_extended[1][1]) ** 2))
    height = int(np.sqrt((keypoints_extended[1][0] - keypoints_extended[2][0]) ** 2 + (keypoints_extended[1][1] - keypoints_extended[2][1]) ** 2))

    # compute new homography matrix for extended area
    x_min = np.min(keypoints_extended[:, 0])
    y_min = np.min(keypoints_extended[:, 1])
    keypoints_extended[:, 0] -= x_min
    keypoints_extended[:, 1] -= y_min
    keypoints_planar = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)
    h, status = cv2.findHomography(keypoints_extended.astype(np.int32), keypoints_planar)
    keypoints_extended[:, 0] += x_min
    keypoints_extended[:, 1] += y_min

    # warp image area
    # extend image if bbox is out of image's plane
    top = 0 if bbox[1] > 0 else -bbox[1]
    bottom = 0 if bbox[3] < im_resized.shape[0] else bbox[3] - im_resized.shape[0]
    left = 0 if bbox[0] > 0 else -bbox[0]
    right = 0 if bbox[2] < im_resized.shape[1] else bbox[2] - im_resized.shape[1]
    im_resized = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    im_dst = cv2.warpPerspective(im_resized[bbox[1] + top:bbox[1] + top + bbox[3], 
                                            bbox[0] + left:bbox[0] + left + bbox[2]], h, (width, height))
    #im_dst = cv2.cvtColor(im_dst, cv2.COLOR_BGR2GRAY)
    #im_dst_eq = cv2.equalizeHist(im_dst)
    return im_dst

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

def export_model_stage1(model_stage_1, labels_stage1):
    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)])
    def serving_fn(image):
        input_image = tf.keras.Input(shape=[None, None, 3], name="image")
        predictions = model_stage_1(input_image, training=False)
        detections = model_stage1.DecodePredictions(confidence_threshold=0.5)(input_image, predictions)
        inference_model = tf.keras.Model(inputs=input_image, outputs=detections)
        bboxes, scores = inference_model(image)
        kept_bboxes, kept_scores = apply_nms(bboxes[0], scores[0])
        kept_bboxes = kept_bboxes.read(0)
        kept_scores = kept_scores.read(0)

        return {
            "bboxes_stage1": kept_bboxes, 
            "scores_stage1": kept_scores, 
            }

    return serving_fn


def export_model_stage2(model_stage_2, labels_stage2):
    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)])
    def serving_fn(image_cropped):
        image = tf.keras.Input(shape=[None, None, 3], name="image_warpped")
        predictions = model_stage_2(image, training=False)
        detections = model_stage2.DecodePredictions(confidence_threshold=0.1)(image, predictions)
        inference_model = tf.keras.Model(inputs=image, outputs=detections)

        detections = inference_model(image_cropped)
        num_detections = detections.valid_detections[0]
        return {
            "bboxes_stage2": detections.nmsed_boxes[0][:num_detections],
            "scores_stage2": detections.nmsed_scores[0][:num_detections]
            }

    return serving_fn


checkpoint_path_stage1 = './retinanet/stage1_reg.keras'
model_stage_1 = tf.keras.saving.load_model(checkpoint_path_stage1)
model_dir = tempfile.gettempdir() + '/models/'
model_sig_version = 1
model_sig_export_path = os.path.join(model_dir, str(model_sig_version))
labels = ['Analog', 'Digital', 'Analog_elligible', 'Digital_elligible']

tf.saved_model.save(
    model_stage_1,
    export_dir=model_sig_export_path,
    signatures={"serving_default": export_model_stage1(model_stage_1, labels)},
)

checkpoint_path_stage2 = 'retinanet/stage2.keras'
model_stage_2 = tf.keras.saving.load_model(checkpoint_path_stage2)
model_dir = tempfile.gettempdir() + '/stage2/'
model_sig_version = 1
model_sig_export_path = os.path.join(model_dir, str(model_sig_version))
labels = ['Analog', 'Digital', 'Analog_elligible', 'Digital_elligible']

tf.saved_model.save(
    model_stage_2,
    export_dir=model_sig_export_path,
    signatures={"serving_default": export_model_stage2(model_stage_2, labels)},
)
