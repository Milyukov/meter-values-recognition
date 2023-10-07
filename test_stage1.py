import os
import cv2
import argparse
import tensorflow_datasets as tfds

from model.data_processing_stage1 import LabelEncoder, preprocess_test_data, resize_and_pad_image
from model.model_stage1 import *
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument("model", help="absolute path to model files", type=str)
parser.add_argument("dataset", help="path to dataset", type=str)
parser.add_argument("binary", help="calc binary classification metrics", type=bool, 
                    default=False, required=False)

class ClassSpecificTruePositives(tf.keras.metrics.Metric):

  def __init__(self, name='class_specific_true_positives', number_of_classes=4, **kwargs):
    super(ClassSpecificTruePositives, self).__init__(name=name, **kwargs)
    self.number_of_classes = number_of_classes
    self.true_positives = {}
    for class_id in self.number_of_classes:
        self.true_positives[class_id] = self.add_weight(name='tp', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    box_labels = y_true[:, :, :12]
    box_predictions = y_pred[:, :, :12]
    cls_predictions = tf.argmax(y_pred[:, :, 12:], axis=-1)


    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
    values = tf.cast(values, self.dtype)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype)
      sample_weight = tf.broadcast_to(sample_weight, values.shape)
      values = tf.multiply(values, sample_weight)
    self.true_positives.assign_add(tf.reduce_sum(values))

  def result(self):
    return self.true_positives

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

def generate_mask(bbox, height, width):
    #points = np.array(np.reshape(bbox[4:], (-1, 2)))
    points = np.array([
        [bbox[4], bbox[5]], 
        [bbox[6], bbox[7]],
        [bbox[8], bbox[9]],
        [bbox[10], bbox[11]]])
    points = points.astype(np.int32)
    mask = cv2.fillPoly(np.zeros((height, width), dtype=np.uint8), 
                        pts = [points], color =(1,))
    return mask

def update_confusion_matrix(confusion_matrix, gt_class, pred_class):
    if gt_class == 0:
        if pred_class == 1:
            confusion_matrix['fp'] += 1
        else:
            confusion_matrix['tn'] += 1
    else:
        if pred_class == 1:
            confusion_matrix['tp'] += 1
        else:
            confusion_matrix['fn'] += 1

if __name__ == '__main__':
    args = parser.parse_args()

    label_encoder = LabelEncoder()
    num_classes = 4
    batch_size = 1

    (test_dataset), dataset_info = tfds.load(
        "meter_values_dataset_stage1", split=["test"], with_info=True, data_dir=args.dataset,
        read_config=tfds.ReadConfig(try_autocache=False)
    )

    test_dataset = test_dataset[0]

    autotune = tf.data.AUTOTUNE

    test_dataset_processed = test_dataset.map(preprocess_test_data, num_parallel_calls=autotune)
    test_dataset_processed = test_dataset_processed.padded_batch(
        batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    test_dataset_processed = test_dataset_processed.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    test_dataset_processed = test_dataset_processed.apply(tf.data.experimental.ignore_errors())
    test_dataset_processed = test_dataset_processed.prefetch(autotune)

    checkpoint_path = args.model
    max_detections = 5

    fn = {i: 0 for i in range(num_classes)}
    fp = {i: 0 for i in range(num_classes)}
    tn = {i: 0 for i in range(num_classes)}
    tp = {i: 0 for i in range(num_classes)}
    confusion_matrix = {'fp': 0, 'tp': 0, 'fn': 0, 'tn': 0}
    thresholds = [0.9]

    if os.path.exists(checkpoint_path):
        model = tf.keras.saving.load_model(checkpoint_path)

        image = tf.keras.Input(shape=[None, None, 3], name="image")
        predictions = model(image, training=False)
        detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
        inference_model = tf.keras.Model(inputs=image, outputs=detections)

        detections = inference_model.predict(test_dataset_processed, batch_size=1)

        # loop over images
        for index, example in enumerate(test_dataset):
            confidences = detections[1][index]
            bboxes = detections[0][index]
            # select only rectangles above a confidence threshold
            valid_detections = np.max(confidences , axis=1) >= 0.5
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
            # loop through example[1] to find all valid GT boxes
            image = example['image']
            height, width, _ = image.shape
            im = image.numpy()
            im, _, ratio = resize_and_pad_image(image, min_side=height, max_side=width, jitter=None)
            print(im.shape)
            im = im.numpy()
            gt_bboxes = example['objects']['bbox']
            gt_labels = example['objects']['label']
            if len(gt_bboxes) == 0:
                if len(kept_bboxes) > 0:
                    # for each class
                    for pred_index, pred_bbox in enumerate(kept_bboxes):
                        # FP += len(bboxes of that class)
                        pred_class = np.argmax(kept_scores[pred_index])
                        if args.binary:
                            pred_class = pred_class % 2
                            update_confusion_matrix(confusion_matrix, gt_class, pred_class)
                        else:
                            fp[pred_class] += 1    
                else:
                    # for each class
                    # TN += 1
                    if args.binary:
                        update_confusion_matrix(confusion_matrix, gt_class, pred_class)
                    else:
                        for i in range(num_classes):
                            tn[i] += 1
                pass
            for gt_index, gt_bbox in enumerate(gt_bboxes):
                gt_class = int(gt_labels[gt_index])
                if args.binary:
                    gt_class = gt_class % 2
                gt_found = False
                if len(kept_bboxes) == 0:
                    # FN += 1
                    if args.binary:
                        update_confusion_matrix(confusion_matrix, gt_class, pred_class)
                    else:
                        fn[gt_class] += 1
                    continue
                # generate mask for gt polygon
                gt_bbox = np.array(gt_bbox)
                gt_bbox[::2] *= height
                gt_bbox[1::2] *= width
                gt_bbox = [
                    gt_bbox[1], 
                    gt_bbox[0],
                    gt_bbox[3], 
                    gt_bbox[2],
                    gt_bbox[5], 
                    gt_bbox[4],
                    gt_bbox[7], 
                    gt_bbox[6],
                    gt_bbox[9], 
                    gt_bbox[8],
                    gt_bbox[11], 
                    gt_bbox[10]
                    ]
                gt_mask = generate_mask(gt_bbox, height, width)
                for pred_index, pred_bbox in enumerate(kept_bboxes):
                    pred_class = np.argmax(kept_scores[pred_index])
                    if args.binary:
                        pred_class = pred_class % 2
                    if gt_class != pred_class:
                        if args.binary:
                            update_confusion_matrix(confusion_matrix, gt_class, pred_class)
                        else:
                            fp[pred_class] += 1
                        continue
                    # generate mask for prediction
                    pred_mask = generate_mask(pred_bbox, height, width)
                    # compute intersection over union
                    iou = np.sum(np.logical_and(gt_mask, pred_mask)) / np.sum(np.logical_or(gt_mask, pred_mask))
                    for thr in thresholds:
                        # if IoU > thr
                        if iou > thr:
                            gt_found = True
                            if args.binary:
                                update_confusion_matrix(confusion_matrix, gt_class, gt_class)
                            else:
                                tp[gt_class] += 1
                        else:
                            if args.binary:
                                update_confusion_matrix(confusion_matrix, gt_class, int(not gt_class))
                            else:
                                fp[gt_class] += 1
                if not gt_found:
                    fn[gt_class] += 1
        if args.binary:
            print(confusion_matrix)
        else:
            for i in range(num_classes):
                print(f'Class # {i}')
                print(f'TP: {tp[i]}, FP: {fp[i]}')
                print(f'FN: {fn[i]}, TN: {tn[i]}')
        
    else:
        print('No checkpoint found')
