import os
import tensorflow_datasets as tfds

from model.data_processing_stage2 import LabelEncoder, preprocess_data, resize_and_pad_image
from model.model_stage2 import *
from model.utils_stage2 import visualize_detections
import tensorflow as tf

import gc
import psutil

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7 # change this value as needed
session = tf.compat.v1.Session(config=config)

class MemoryUsageCallbackExtended(tf.keras.callbacks.Callback):
  '''Monitor memory usage on epoch begin and end, collect garbage'''

  def on_epoch_begin(self,epoch,logs=None):
    print('**Epoch {}**'.format(epoch))
    print('Memory usage on epoch begin: {}'.format(psutil.Process(os.getpid()).memory_info().rss))

  def on_epoch_end(self,epoch,logs=None):
    print('Memory usage on epoch end:   {}'.format(psutil.Process(os.getpid()).memory_info().rss))
    gc.collect()
    tf.keras.backend.clear_session()

model_dir = "retinanet/"
label_encoder = LabelEncoder()

num_classes = 17
batch_size = 16

learning_rates = [0.001, 0.00005]#[2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [20]#[125, 250,500, 240000, 360000] 
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss=loss_fn, optimizer=optimizer)#, run_eagerly=True)

checkpoint_path = "retinanet/stage2.keras"
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="loss",
        save_best_only=False,
        save_weights_only=False,
        save_freq=1,
        verbose=1,
    ),
    MemoryUsageCallbackExtended()
]

(train_dataset, val_dataset), dataset_info = tfds.load(
    "meter_values_dataset_stage2", split=["train", "validation"], with_info=True,
    read_config=tfds.ReadConfig(try_autocache=False), data_dir="/home/gleb/Projects/counters-datasets/meter_values_dataset_stage2"
)

autotune = tf.data.AUTOTUNE

train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
train_dataset = train_dataset.shuffle(batch_size)
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
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)

# Uncomment the following lines, when training on full dataset
train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
print(f'Train steps per epoch = {train_steps_per_epoch}')
val_steps_per_epoch = \
    dataset_info.splits["validation"].num_examples // batch_size

train_steps = 4 * 100000
epochs = train_steps // train_steps_per_epoch

import os
if os.path.exists(checkpoint_path):
    model = tf.keras.saving.load_model(checkpoint_path)

model.fit(
    train_dataset.take,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)


def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

int2str = dataset_info.features["objects"]["label"].int2str

for sample in val_dataset.take(2):
    image = tf.cast(sample, dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    print(input_image.shape)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )
