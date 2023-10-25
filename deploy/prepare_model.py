# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors,
# and stored with the default serving key
import tensorflow as tf
import os
import sys

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__)) + '/../'

# Add the parent directory to sys.path
sys.path.append(parent_dir)
sys.path.append(parent_dir + 'model')

# Import the module from the parent directory
from model import model_stage1, data_processing_stage1
from model import model_stage2, data_processing_stage2

def apply_nms(bboxes, confidences, confidence_thr=0.0):
    # select only rectangles above a confidence threshold
    max_confidences = tf.reduce_max(confidences , axis=-1, keepdims=True)
    valid_detections = max_confidences >= confidence_thr
    confidences = confidences[valid_detections[:, :, 0]]
    bboxes = bboxes[valid_detections[:, :, 0]]
    # sort the thresholded rectangles in descending order
    sorted_indices = tf.argsort(max_confidences, direction='DESCENDING', axis=1)
    sorted_scores = tf.gather(confidences, sorted_indices[:, :, 0])
    sorted_bboxes = tf.gather(bboxes, sorted_indices[:, :, 0])
    # reshape tensors and return the one with the highest confidence
    sorted_bboxes = tf.reshape(sorted_bboxes, [-1, 12])
    sorted_scores = tf.reshape(sorted_scores, [-1, 4])
    return sorted_bboxes[0, :], sorted_scores[0, :]

def export_model_stage1(model_stage_1, labels_stage1):
    @tf.function(input_signature=[tf.TensorSpec([None, None, 3], tf.float32)])
    def prepare_image(image):
        image = tf.cast(image, dtype=tf.float32)
        resized_image, _, ratio = data_processing_stage1.resize_and_pad_image(image, jitter=None)
        processed_image = tf.keras.applications.resnet.preprocess_input(resized_image)
        processed_image = tf.expand_dims(processed_image, axis=0)
        return processed_image, resized_image

    @tf.function(input_signature=[tf.TensorSpec([None, None, 3], tf.float32)])
    def serving_fn(image):
        input_image = tf.keras.Input(shape=[None, None, 3], name="image")
        predictions = model_stage_1(input_image, training=False)
        detections = model_stage1.DecodePredictions(confidence_threshold=0.5)(input_image, predictions)
        inference_model = tf.keras.Model(inputs=input_image, outputs=detections)

        processed_image, resized_image = prepare_image(image)
        bboxes, scores = inference_model(processed_image)
        kept_bboxes, kept_scores = apply_nms(bboxes, scores, confidence_thr=0.0)
        kept_bboxes = tf.expand_dims(kept_bboxes, axis=0)
        kept_scores = tf.expand_dims(kept_scores, axis=0)
        resized_image = tf.expand_dims(resized_image, axis=0)
        return {
            "bboxes_stage1": kept_bboxes, 
            "scores_stage1": kept_scores,
            "resized_image": resized_image
            }

    return serving_fn


def export_model_stage2(model_stage_2, labels_stage2):
    @tf.function(input_signature=[tf.TensorSpec([None, None, 3], tf.float32)])
    def prepare_image(image):
        processed_image, _, ratio = data_processing_stage2.resize_and_pad_image(image, jitter=None)
        processed_image = tf.keras.applications.resnet.preprocess_input(processed_image)
        processed_image = tf.expand_dims(processed_image, axis=0)
        return processed_image, ratio
    
    @tf.function(input_signature=[tf.TensorSpec([None, None, 3], tf.float32)])
    def serving_fn(image_cropped):
        image = tf.keras.Input(shape=[None, None, 3], name="image_warpped")
        predictions = model_stage_2(image, training=False)
        detections = model_stage2.DecodePredictions(confidence_threshold=0.5)(image, predictions)
        inference_model = tf.keras.Model(inputs=image, outputs=detections)
        processed_image, ratio = prepare_image(image_cropped)
        detections = inference_model(processed_image)
        num_detections = detections.valid_detections[0]
        kept_bboxes = detections.nmsed_boxes[0][:num_detections]
        kept_scores = detections.nmsed_scores[0][:num_detections]
        # raw_scores = detections.nmsed_scores
        labels = detections.nmsed_classes[0][:num_detections]
        kept_bboxes = tf.expand_dims(kept_bboxes, axis=0)
        kept_scores = tf.expand_dims(kept_scores, axis=0)
        labels = tf.expand_dims(labels, axis=0)
        ratio = tf.expand_dims(ratio, axis=0)
        # raw_scores = tf.expand_dims(ratio, axis=0)
        return {
            "bboxes_stage2": kept_bboxes,
            "scores_stage2": kept_scores,
            "labels": labels,
            "ratio": ratio,
            # "raw_scores": raw_scores
            }

    return serving_fn

if __name__ == '__main__':
    checkpoint_path_stage1 = './retinanet/stage1_reg.keras'
    model_stage_1 = tf.keras.saving.load_model(checkpoint_path_stage1)
    model_dir = './models/stage1/'
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
    model_dir = './models/stage2/'
    model_sig_version = 1
    model_sig_export_path = os.path.join(model_dir, str(model_sig_version))
    labels = [f'{i}' for i in range(17)]

    tf.saved_model.save(
        model_stage_2,
        export_dir=model_sig_export_path,
        signatures={"serving_default": export_model_stage2(model_stage_2, labels)},
    )
