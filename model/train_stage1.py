from datetime import datetime
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

import matplotlib.pyplot as plt

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
    batch_size = 8

    learning_rates = [0.01, 0.001, 0.0005]#[2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [1, 10]#[125, 250,500, 240000, 360000] 
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, resnet50_backbone)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=loss_fn, optimizer=optimizer, run_eagerly=True)

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    checkpoint_path = "retinanet/stage1.keras"
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="loss",
            save_best_only=False,
            save_weights_only=False,
            save_freq=10,
            verbose=1,
        ),
        MemoryUsageCallbackExtended(), 
        tensorboard_callback
    ]

    (train_dataset, val_dataset), dataset_info = tfds.load(
        "meter_values_dataset_stage1", split=["train", "test"], with_info=True, data_dir="/home/gleb/tensorflow_datasets",
        read_config=tfds.ReadConfig(try_autocache=False)
    )

    autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
    train_dataset = train_dataset.shuffle(batch_size)
    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    train_dataset = train_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    )
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(autotune)

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
        dataset_info.splits["test"].num_examples // batch_size

    train_steps = 4 * 100000
    epochs = train_steps // train_steps_per_epoch

    print(f'Number of epochs = {epochs}')

    if os.path.exists(checkpoint_path):
        model = tf.keras.saving.load_model(checkpoint_path)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1
    )
