import os
import argparse
import numpy as np
import tensorflow_datasets as tfds
import time

from model.data_processing_stage1 import LabelEncoder, resize_and_pad_image
from model.model_stage1 import *
from model.utils_stage1 import visualize_detections

parser = argparse.ArgumentParser()
parser.add_argument("model", help="absolute path to model files", type=str)
parser.add_argument("images", help="path to images", type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # change this value as needed
    session = tf.compat.v1.Session(config=config)

    label_encoder = LabelEncoder()
    num_classes = 4
    batch_size = 1
    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, resnet50_backbone)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.000005)#learning_rate_fn)
    model.compile(loss=loss_fn, optimizer=optimizer)#, run_eagerly=True)

    checkpoint_path = args.model
    if os.path.exists(checkpoint_path):
        model = tf.keras.saving.load_model(checkpoint_path)

    def prepare_image(image):
        image, _, ratio = resize_and_pad_image(image, jitter=None)
        image = tf.keras.applications.resnet.preprocess_input(image)
        return tf.expand_dims(image, axis=0), ratio

    image = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = model(image, training=False)
    detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
    inference_model = tf.keras.Model(inputs=image, outputs=detections)

    # val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
    # int2str = dataset_info.features["objects"]["label"].int2str

    def get_iou(ground_truth, pred):
        # coordinates of the area of intersection.
        ix1 = np.maximum(ground_truth[0], pred[0])
        iy1 = np.maximum(ground_truth[1], pred[1])
        ix2 = np.minimum(ground_truth[2], pred[2])
        iy2 = np.minimum(ground_truth[3], pred[3])
        
        # Intersection height and width.
        i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
        i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
        
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

    max_detections = 5

    image_loaded = cv2.imread('image_loaded_2.jpg')

    for filename in os.listdir(args.images):
        if not filename.lower().endswith(('.jpg', '*.png', '*.bmp')):
           continue
        st = time.time()
        image = cv2.imread(os.path.join(args.images, filename))
        image = tf.cast(image, dtype=tf.float32)
        input_image, ratio = prepare_image(image)
        detections = inference_model.predict(input_image)

        # loop over images
        for index, confidences in enumerate(detections[1]):
           bboxes = detections[0][index]
           # select only rectangles above a confidence threshold
           valid_detections = np.max(confidences , axis=1) >= 0.0
           confidences = confidences[valid_detections]
           bboxes = bboxes[valid_detections]
           # sort the thresholded rectangles in descending order
           sorted_indices = sorted(range(len(confidences)),key=lambda index: np.max(confidences[index]), reverse=True)
           sorted_scores = confidences[sorted_indices]
           sorted_bboxes = bboxes[sorted_indices]
           # create an empty set of kept rectangle
           kept_bboxes = []
           kept_scores = []
           # loop over the sorted thresholded rectangles
           max_detections_val = min(max_detections, len(sorted_indices))
           for bbox_index, bbox in enumerate(sorted_bboxes[:max_detections_val]):
            # loop over the set of kept rectangles:
            all_iou_lower = True
            for kept_bbox in kept_bboxes:
                # compute IOU between the rectangles
                iou = get_iou(bbox, kept_bbox)
                # if IOU is above IOU threshold break loop
                if iou > 0.7:
                   all_iou_lower = False
                   break
            # if all IOU are below the IOU threshold add to kept
            if all_iou_lower:
               kept_bboxes.append(bbox)
               kept_scores.append(sorted_scores[bbox_index])

        et = time.time()
        print(f'Inference time = {et - st}')

        class_names= ['Analog', 'Digital', 'Analog_elligible', 'Digital_elligible']
        visualize_detections(
            image,
            map(lambda x: x / ratio, kept_bboxes),
            class_names,
            kept_scores
        )
