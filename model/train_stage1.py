import os
import numpy as np
import zipfile
import tensorflow_datasets as tfds

from data_processing_stage1 import LabelEncoder, preprocess_data, resize_and_pad_image
from model_stage1 import *
from utils_stage1 import visualize_detections
from tensorflow import keras

import gc
import psutil

class MemoryUsageCallbackExtended(tf.keras.callbacks.Callback):
  '''Monitor memory usage on epoch begin and end, collect garbage'''

  def on_epoch_begin(self,epoch,logs=None):
    print('**Epoch {}**'.format(epoch))
    print('Memory usage on epoch begin: {}'.format(psutil.Process(os.getpid()).memory_info().rss))

  def on_epoch_end(self,epoch,logs=None):
    print('Memory usage on epoch end:   {}'.format(psutil.Process(os.getpid()).memory_info().rss))
    gc.collect()
    tf.keras.backend.clear_session()

def download_extract_dataset(url):
    try:
        filename = os.path.join(os.getcwd(), "data.zip")
        keras.utils.get_file(filename, url)

        with zipfile.ZipFile("data.zip", "r") as z_fp:
            z_fp.extractall("./")
        return True
    except:
       return False

if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # change this value as needed
    session = tf.compat.v1.Session(config=config)

    model_dir = "retinanet/"
    label_encoder = LabelEncoder()

    num_classes = 4
    batch_size = 1

    learning_rates = [0.001, 0.0001]#[2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [200]#[125, 250,500, 240000, 360000] 
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, resnet50_backbone)

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    model.compile(loss=loss_fn, optimizer=optimizer)#, run_eagerly=True)

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            #filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            filepath=os.path.join(model_dir, "weights_best"),
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
            #save_freq=100,
            verbose=1,
        ),
        MemoryUsageCallbackExtended()
    ]

    (train_dataset, val_dataset), dataset_info = tfds.load(
        "meter_values_dataset", split=["train", "test"], with_info=True, data_dir="/home/gleb/tensorflow_datasets",
        read_config=tfds.ReadConfig(try_autocache=False)
    )

    autotune = tf.data.AUTOTUNE
    #train_dataset = train_dataset[0]

    for example in train_dataset:
        image_loaded = example['image'].numpy()
        break

    train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=1)#autotune)
    # train_dataset = train_dataset.shuffle(batch_size)
    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    train_dataset = train_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=1#autotune
    )
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(0)

    val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
    val_dataset = val_dataset.padded_batch(
        batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=1)#autotune)
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    val_dataset = val_dataset.prefetch(0)

    # Uncomment the following lines, when training on full dataset
    # train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
    # print(f'Train steps per epoch = {train_steps_per_epoch}')
    # val_steps_per_epoch = \
    #     dataset_info.splits["validation"].num_examples // batch_size

    # train_steps = 4 * 100000
    # epochs = train_steps // train_steps_per_epoch

    epochs = 10

    # Running 1 training step,
    # remove `.take` when training on the full dataset

    model.fit(
        train_dataset.take(1),
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )

    model.save('./data/model.keras')

    def prepare_image(image):
        image, _, ratio = resize_and_pad_image(image, jitter=None)
        image = tf.keras.applications.resnet.preprocess_input(image)
        return tf.expand_dims(image, axis=0), ratio

    # Change this to `model_dir` when not using the downloaded weights
    # weights_dir = "data"
    # latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
    # model.load_weights(latest_checkpoint)
    # model = keras.models.load_model('./data/model.keras')

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

    max_detections = 100

    for sample in [image_loaded]:#val_dataset.take(2):
        image = tf.cast(sample, dtype=tf.float32)
        input_image, ratio = prepare_image(image)
        detections = inference_model.predict(input_image)

        # loop over images
        for index, confidences in enumerate(detections[1]):
           bboxes = detections[0][index]
           # select only rectangles above a confidence threshold
           valid_detections = np.max(confidences , axis=1) > 0.001
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

        #num_detections = detections.valid_detections[0]
        # class_names = [
        #     int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
        # ]
        class_names= []
        visualize_detections(
            image,
            map(lambda x: x / ratio, kept_bboxes),#detections.nmsed_boxes[0][:num_detections] / ratio,
            class_names,
            kept_scores#detections.nmsed_scores[0][:num_detections],
        )
