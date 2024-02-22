from flask import Flask, request, jsonify
import traceback
import os
from collections import OrderedDict

import tensorflow as tf
import numpy as np

from utils import *

import time

from official.vision.ops.preprocess_ops import resize_and_crop_image

HOST = '0.0.0.0'
PORT_NUMBER = 8080

parent_dir = os.path.dirname(os.path.realpath(__file__))

def build_inputs_for_object_detection(image, input_image_size):
  """Builds Object Detection model inputs for serving."""
  image, _ = resize_and_crop_image(
      image,
      input_image_size,
      padded_size=input_image_size,
      aug_scale_min=1.0,
      aug_scale_max=1.0)
  return image

class MeterValuesRecognition:

    def __init__(self):
        self.saved_path = os.path.join(parent_dir, 'models')
        self.stage1 = tf.saved_model.load(os.path.join(self.saved_path, 'stage1/1'))
        self.stage2_analog = tf.saved_model.load(os.path.join(self.saved_path, 'stage2_analog/1'))
        self.stage2_digital_electrical = tf.saved_model.load(os.path.join(self.saved_path, 'exported_model_counters'))
        self.stage2_digital_water = tf.saved_model.load(os.path.join(self.saved_path, 'stage2_digital/1'))

        self.predict_stage1 = self.stage1.signatures["serving_default"]
        self.predict_stage2_analog = self.stage2_analog.signatures["serving_default"]
        self.predict_stage2_digital_electrical = self.stage2_digital_electrical.signatures["serving_default"]
        self.predict_stage2_digital_water = self.stage2_digital_water.signatures["serving_default"]
        self.int2label_stage1 ={
            0: "analog",
            1: "digital",
            2: "analog_illegible",
            3: "digital_illegible"
        }
        self.id2str = OrderedDict([
      (0, '0'),
      (1, '1'),
      (2, '2'),
      (3, '3'),
      (4, '4'),
      (5, '5'),
      (6, '6'),
      (7, '7'),
      (8, '8'),
      (9, '9'),
      (10, 'R'),
      (11, 'T'),
      (12, 'M'),
      (13, '_'),
      (14, 'floatp'),
      (15, ':'),
      (16, '^'),
      (17, 'Q'),
      (18, 'V'),
      (19, 'U'),
      (20, '+'),
      (21, '-'),
      (22, 'Ч'),
      (23, 'C')
  ])
        self.id2str_tf={
    0: {
            "id": 0,
            "name": "+"
        },
    1:  {
            "id": 1,
            "name": "-"
        },
    2:  {
            "id": 2,
            "name": "0"
        },
    3:  {
            "id": 3,
            "name": "1"
        },
    4:  {
            "id": 4,
            "name": "2"
        },
    5:  {
            "id": 5,
            "name": "3"
        },
    6:  {
            "id": 6,
            "name": "4"
        },
    7:  {
            "id": 7,
            "name": "5"
        },
    8:  {
            "id": 8,
            "name": "6"
        },
    9:  {
            "id": 9,
            "name": "7"
        },
    10: {
            "id": 10,
            "name": "8"
        },
    11: {
            "id": 11,
            "name": "9"
        },
    12: {
            "id": 12,
            "name": "COLON"
        },
    13: {
            "id": 13,
            "name": "FLOATP"
        },
    14: {
            "id": 14,
            "name": "M"
        },
    15: {
            "id": 15,
            "name": "Q"
        },
    16: {
            "id": 16,
            "name": "R"
        },
    17: {
            "id": 17,
            "name": "T"
        },
    18: {
            "id": 18,
            "name": "U"
        },
    19: {
            "id": 19,
            "name": "V"
        },
    20: {
            "id": 20,
            "name": "_"
        },
    21: {
            "id": 21,
            "name": "arrow"
        },
    22: {
            "id": 22,
            "name": "Аналоговый счётчик"
        },
    23: {
            "id": 23,
            "name": "С"
        },
    24: {
            "id": 24,
            "name": "Цифровой счётчик"
        },
    25: {
            "id": 25,
            "name": "Ч"
        }
}

    def infer(self, image, counter_type, vis=False):
        st = time.time()
        image = np.array(image, dtype=np.uint8)
        response = {
            "original_image_shape": image.shape,
            "success": False,
            "text": "",
            "counter_class": "",
            "counter_score": 0.0,
            "time": 0.0
        }
        tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
        predictions = self.predict_stage1(tensor_image)
        kept_bboxes = np.array(predictions['bboxes_stage1'])[0]
        kept_scores = np.array(predictions['scores_stage1'])[0]
        image_resized = np.array(predictions['resized_image'])[0]

        detected_class = tf.argmax(kept_scores).numpy()
        detected_score = kept_scores[detected_class]
        if detected_score > 0.5:
            response["counter_class"] = self.int2label_stage1[detected_class]
        else:
            response["counter_class"] = 'other'
        response["counter_score"] = str(detected_score)
        # if detected_class > 1:
        #     et = time.time()
        #     response["time"] = str(et - st)
        #     return response            
        try:
            if detected_class in [0, 2]:
                image_cropped, roi = extract_rectangle_area(image_resized, kept_bboxes[:4], kept_bboxes[4:])
            else:
                image_cropped, roi = extract_rectangle_area(
                    image_resized, kept_bboxes[:4], kept_bboxes[4:])#, x_extend=0, y_extend=0)
        except:
            response["error"] = "error in warping after 1st stage"
            return response

        if detected_class in [0, 2]:
            tensor_image = tf.convert_to_tensor(image_cropped, dtype=tf.float32)
            predictions = self.predict_stage2_analog(tensor_image)
            kept_bboxes = np.array(predictions['bboxes_stage2'])[0]
            kept_scores = np.array(predictions['scores_stage2'])[0]
            labels = np.array(predictions['labels'])[0]
            ratio = np.array(predictions['ratio'])[0]
            class_names= [f'{self.id2str[int(x)]}' for x in labels]
        else:
            if counter_type == 'electrical':
                ratio =  512 / np.max(image_cropped.shape[:2])
                image = build_inputs_for_object_detection(image_cropped, (512, 512))
                tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
                tensor_image = tf.expand_dims(tensor_image, axis=0)
                tensor_image = tf.cast(tensor_image, dtype = tf.uint8)
                predictions = self.predict_stage2_digital_electrical(tensor_image)
                kept_bboxes = np.array(predictions['detection_boxes'])[0][:, (1, 0, 3, 2)]
                kept_scores = np.array(predictions['detection_scores'])[0]
                labels = np.array(predictions['detection_classes'])[0]
                indices = kept_scores > 0.4
                kept_bboxes = kept_bboxes[indices]
                kept_scores = kept_scores[indices]
                labels = labels[indices]
                class_names= [f'{self.id2str_tf[int(x)]["name"]}' for x in labels]
            else:
                tensor_image = tf.convert_to_tensor(image_cropped, dtype=tf.float32)
                predictions = self.predict_stage2_digital_water(tensor_image)
                kept_bboxes = np.array(predictions['bboxes_stage2'])[0]
                kept_scores = np.array(predictions['scores_stage2'])[0]
                labels = np.array(predictions['labels'])[0]
                ratio = np.array(predictions['ratio'])[0]
                class_names= [f'{self.id2str[int(x)]}' for x in labels]

        response['roi'] = roi.tolist()
        roi = roi.astype(np.float64) * ratio

        if detected_class in [0, 2]:
            text, boxes, scores, class_names = parse_analog_detection(kept_bboxes, kept_scores, class_names, roi)
        else:
            text, boxes, scores, class_names = parse_digital_detection(kept_bboxes, kept_scores, class_names, roi)
        if not 'x' in text:
            response["success"] = True
            response["counter_class"].replace('_illegible', '')
        response["text"] = text
        et = time.time()
        response["time"] = str(et - st)

        # for visualization
        if vis:
            response["bboxes"] = boxes.tolist()
            response["scores"] = scores.tolist()
            response["class_names"] = class_names.tolist()
            response["ratio"] = ratio.tolist()
            response["image_cropped"] = image_cropped.tolist()
        return response

if __name__ == '__main__':
    os.system('nvidia-smi')

    app = Flask(__name__)
    ocr = MeterValuesRecognition()

    @app.route('/recognize', methods=["POST"])
    def infer():
        image_file = request.files['image']
        image = cv2.imdecode(np.fromfile(image_file, np.uint8), cv2.IMREAD_UNCHANGED)
        return ocr.infer(image, True)


    @app.route('/recognizeJSON', methods=["POST"])
    def infer_json():
        data = request.json
        image = data['image']
        counter_type = data.get('counter_type', 'electrical')
        return ocr.infer(image, counter_type, False)


    @app.errorhandler(Exception)
    def handle_exception(e):
        return jsonify(stackTrace=traceback.format_exc())

    app.run(host=HOST, port=PORT_NUMBER)
