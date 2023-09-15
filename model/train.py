import os
import numpy as np
import zipfile
import tensorflow_datasets as tfds

from data_processing import LabelEncoder, preprocess_data, resize_and_pad_image
from model import *
from utils import visualize_detections
from tensorflow import keras

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
    #tf.keras.backend.clear_session()


# url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
# filename = os.path.join(os.getcwd(), "data.zip")
# keras.utils.get_file(filename, url)


# with zipfile.ZipFile("data.zip", "r") as z_fp:
#     z_fp.extractall("./")

model_dir = "retinanet/"
label_encoder = LabelEncoder()

num_classes = 4
batch_size = 1

learning_rates = [0.001, 0.0001, 0.00001]#[2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250]#[125, 250,500, 240000, 360000] 
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)#, run_eagerly=True)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        save_freq=1010,
        verbose=1,
    ),
    MemoryUsageCallbackExtended()
]

train_dataset, dataset_info = tfds.load(
    "meter_values_dataset", split=["train"], with_info=True, data_dir="/home/gleb/tensorflow_datasets",
    read_config=tfds.ReadConfig(try_autocache=False)
)

# autotune = tf.data.AUTOTUNE
train_dataset = train_dataset[0]

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

# val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
# val_dataset = val_dataset.padded_batch(
#     batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
# )
# val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=1)#autotune)
# val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
#val_dataset = val_dataset.prefetch(autotune)

for example in train_dataset:
    x = example[0].numpy()
    y = example[1].numpy()
    break

# Uncomment the following lines, when training on full dataset
train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
print(f'Train steps per epoch = {train_steps_per_epoch}')
# val_steps_per_epoch = \
#     dataset_info.splits["validation"].num_examples // batch_size

# train_steps = 4 * 100000
# epochs = train_steps // train_steps_per_epoch

epochs = 500

# Running 100 training and 50 validation steps,
# remove `.take` when training on the full dataset

model.fit(
    train_dataset.take(1),
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)


def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

# for epoch in range(epochs):
#     print('Memory usage on epoch begin: {}'.format(psutil.Process(os.getpid()).memory_info().rss))

#     # Iterate over the batches of the dataset.
#     for step, example in enumerate(train_dataset):
#         images, bboxes, cls = preprocess_data(example)
#         images = tf.expand_dims(images, 0)
#         bboxes = tf.expand_dims(bboxes, 0)
#         cls = tf.expand_dims(cls, 0)
#         x_batch_train, y_batch_train = label_encoder.encode_batch(images, bboxes, cls)
#         # Open a GradientTape to record the operations run
#         # during the forward pass, which enables auto-differentiation.
#         with tf.GradientTape() as tape:
#             # Run the forward pass of the layer.
#             # The operations that the layer applies
#             # to its inputs are going to be recorded
#             # on the GradientTape.
#             predictions = model(x_batch_train, training=True)  # Logits for this minibatch

#             # Compute the loss value for this minibatch.
#             loss_value = loss_fn(y_batch_train, predictions)

#         # Use the gradient tape to automatically retrieve
#         # the gradients of the trainable variables with respect to the loss.
#         grads = tape.gradient(loss_value, model.trainable_weights)

#         # Run one step of gradient descent by updating
#         # the value of the variables to minimize the loss.
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))

#         print('Memory usage on epoch end:   {}'.format(psutil.Process(os.getpid()).memory_info().rss))

#         # Log every 200 batches.
#         if step % 200 == 0:
#             print(
#                 "Training loss (for one batch) at step %d: %.4f"
#                 % (step, float(loss_value))
#             )
#             print("Seen so far: %s samples" % ((step + 1) * batch_size))

# model.save('./data/model.keras')

# Change this to `model_dir` when not using the downloaded weights
#weights_dir = "data"
weights_dir = "./data/"

#latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
#model.load_weights(latest_checkpoint)
# model = keras.models.load_model('./data/model.keras')

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)




# val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
# int2str = dataset_info.features["objects"]["label"].int2str

for sample in [image_loaded]:#val_dataset.take(2):
    image = tf.cast(sample, dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    #assert input_image - x
    print(input_image.shape)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    # class_names = [
    #     int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    # ]
    class_names= []
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )
