import pprint
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_models as tfm

import matplotlib

from official.core import exp_factory
from official.vision.utils.object_detection import visualization_utils
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder

pp = pprint.PrettyPrinter(indent=4) # Set Pretty Print Indentation
print(tf.__version__) # Check the version of tensorflow used

train_data_input_path = './counters/train-00000-of-00001.tfrecord'
valid_data_input_path = './counters/validation-00000-of-00001.tfrecord'
model_dir = './trained_model_counters/'

exp_config = exp_factory.get_exp_config('retinanet_resnetfpn_coco')

batch_size = 2
num_classes = 26

HEIGHT, WIDTH = 512, 512
IMG_SIZE = [HEIGHT, WIDTH, 3]

# Backbone config.
exp_config.task.freeze_backbone = False
exp_config.task.annotation_file = ''

# Model config.
exp_config.task.model.input_size = IMG_SIZE
exp_config.task.model.num_classes = num_classes + 1
exp_config.task.model.detection_generator.tflite_post_processing.max_classes_per_detection = exp_config.task.model.num_classes

# Training data config.
exp_config.task.train_data.input_path = train_data_input_path
exp_config.task.train_data.dtype = 'float32'
exp_config.task.train_data.global_batch_size = batch_size
exp_config.task.train_data.parser.aug_scale_max = 1.0
exp_config.task.train_data.parser.aug_scale_min = 1.0

# Validation data config.
exp_config.task.validation_data.input_path = valid_data_input_path
exp_config.task.validation_data.dtype = 'float32'
exp_config.task.validation_data.global_batch_size = batch_size

logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]

if 'GPU' in ''.join(logical_device_names):
  print('This may be broken in Colab.')
  device = 'GPU'
elif 'TPU' in ''.join(logical_device_names):
  print('This may be broken in Colab.')
  device = 'TPU'
else:
  print('Running on CPU is slow, so only train for a few steps.')
  device = 'CPU'


train_steps = 50000
exp_config.trainer.steps_per_loop = 100 # steps_per_loop = num_of_training_examples // train_batch_size

exp_config.trainer.summary_interval = 100
exp_config.trainer.checkpoint_interval = 100
exp_config.trainer.validation_interval = 5000
exp_config.trainer.validation_steps =  400 # validation_steps = num_of_validation_examples // eval_batch_size
exp_config.trainer.train_steps = train_steps
exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 100
exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.1
exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.05

pp.pprint(exp_config.as_dict())

if exp_config.runtime.mixed_precision_dtype == tf.float16:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

if 'GPU' in ''.join(logical_device_names):
  distribution_strategy = tf.distribute.MirroredStrategy()
elif 'TPU' in ''.join(logical_device_names):
  tf.tpu.experimental.initialize_tpu_system()
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='/device:TPU_SYSTEM:0')
  distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
  print('Warning: this will be really slow.')
  distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])

print('Done')

with distribution_strategy.scope():
  task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir)

for images, labels in task.build_inputs(exp_config.task.train_data).take(1):
  print()
  print(f'images.shape: {str(images.shape):16}  images.dtype: {images.dtype!r}')
  print(f'labels.keys: {labels.keys()}')

category_index={
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
tf_ex_decoder = TfExampleDecoder()

def show_batch(raw_records, num_of_examples):
  plt.figure(figsize=(20, 20))
  use_normalized_coordinates=True
  min_score_thresh = 0.30
  for i, serialized_example in enumerate(raw_records):
    plt.subplot(1, num_of_examples, i + 1)
    decoded_tensors = tf_ex_decoder.decode(serialized_example)
    image = decoded_tensors['image'].numpy().astype('uint8')
    scores = np.ones(shape=(len(decoded_tensors['groundtruth_boxes'])))
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image,
        decoded_tensors['groundtruth_boxes'].numpy(),
        decoded_tensors['groundtruth_classes'].numpy().astype('int'),
        scores,
        category_index=category_index,
        use_normalized_coordinates=use_normalized_coordinates,
        max_boxes_to_draw=200,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        instance_masks=None,
        line_thickness=4)

    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Image-{i+1}')
  plt.show()

buffer_size = 20
num_of_examples = 6

raw_records = tf.data.TFRecordDataset(
    exp_config.task.train_data.input_path).shuffle(
        buffer_size=buffer_size).take(num_of_examples)
matplotlib.use('TKAgg', force=True)
print("Using:",matplotlib.get_backend())
# show_batch(raw_records, num_of_examples)

model, eval_logs = tfm.core.train_lib.run_experiment(
    distribution_strategy=distribution_strategy,
    task=task,
    mode='train_and_eval',
    params=exp_config,
    model_dir=model_dir,
    run_post_eval=True)
