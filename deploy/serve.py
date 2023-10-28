from flask import Flask, request, jsonify
import traceback
import os

import tensorflow as tf
import numpy as np

from utils import extract_rectangle_area, parse_analog_detection

import time

HOST = '0.0.0.0'
PORT_NUMBER = 8080

parent_dir = os.path.dirname(os.path.realpath(__file__))

class MeterValuesRecognition:

    def __init__(self):
        self.saved_path = os.path.join(parent_dir, 'models')
        self.stage1 = tf.saved_model.load(os.path.join(self.saved_path, 'stage1/1'))
        self.stage2 = tf.saved_model.load(os.path.join(self.saved_path, 'stage2/1'))

        self.predict_stage1 = self.stage1.signatures["serving_default"]
        self.predict_stage2 = self.stage2.signatures["serving_default"]
        self.int2label_stage1 ={
            0: "analog",
            1: "digital",
            2: "analog_illegible",
            3: "digital_illegible"
        }

    def infer(self, image, vis=False):
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
        response["counter_class"] = self.int2label_stage1[detected_class]
        response["counter_score"] = str(detected_score)
        if detected_class % 2 == 1 or (detected_class == 0 and detected_score <= 0.5) or (detected_class == 2 and detected_score >= 0.5):
            et = time.time()
            response["time"] = str(et - st)
            return response            
        try:
            image_cropped, roi = extract_rectangle_area(image_resized, kept_bboxes[:4], kept_bboxes[4:])
        except:
            response["error"] = "error in warping after 1st stage"
            return response
        response['roi'] = roi.tolist()
        tensor_image = tf.convert_to_tensor(image_cropped, dtype=tf.float32)
        predictions = self.predict_stage2(tensor_image)
        kept_bboxes = np.array(predictions['bboxes_stage2'])[0]
        kept_scores = np.array(predictions['scores_stage2'])[0]
        labels = np.array(predictions['labels'])[0]
        ratio = np.array(predictions['ratio'])[0]

        class_names= [f'{int(x)}' for x in labels]

        text, boxes, scores, class_names = parse_analog_detection(kept_bboxes, kept_scores, class_names, roi)
        if not 'x' in text:
            response["success"] = True
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
        data = request.json
        image = data['image']
        return ocr.infer(image, True)
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        return jsonify(stackTrace=traceback.format_exc())

    app.run(host=HOST, port=PORT_NUMBER)
