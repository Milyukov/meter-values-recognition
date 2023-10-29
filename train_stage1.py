from datetime import datetime
import os
import tensorflow_datasets as tfds

from model.data_processing_stage1 import LabelEncoder, preprocess_data
from model.model_stage1 import *
from tensorflow import keras

import gc
import psutil

import argparse

class MemoryUsageCallbackExtended(tf.keras.callbacks.Callback):
  '''Monitor memory usage on epoch begin and end, collect garbage'''

  def on_epoch_begin(self,epoch,logs=None):
    print('**Epoch {}**'.format(epoch))
    print('Memory usage on epoch begin: {}'.format(psutil.Process(os.getpid()).memory_info().rss))

  def on_epoch_end(self,epoch,logs=None):
    print('Memory usage on epoch end:   {}'.format(psutil.Process(os.getpid()).memory_info().rss))
    gc.collect()
    tf.keras.backend.clear_session()


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", help="path to dataset", type=str, 
                    default='/mnt/images/counters-datasets/meter_values_dataset_stage1')
parser.add_argument("--resume_training", help="Boolean flag wheather to resume training if it was interrupted", 
                    action="store_true")

if __name__ == '__main__':
    args = parser.parse_args()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # change this value as needed
    session = tf.compat.v1.Session(config=config)

    model_dir = "retinanet/"
    label_encoder = LabelEncoder()

    num_classes = 4
    batch_size = 8

    learning_rates = [0.0001, 0.00001, 0.000005]#[2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [23, 50]#[125, 250,500, 240000, 360000] 
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, resnet50_backbone)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    model.compile(loss=loss_fn, optimizer=optimizer, run_eagerly=False)

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
        "meter_values_dataset_stage1", split=["train", "validation"], with_info=True, 
        data_dir=args.dataset_path, read_config=tfds.ReadConfig(try_autocache=False)
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

    train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
    print(f'Train steps per epoch = {train_steps_per_epoch}')
    val_steps_per_epoch = \
        dataset_info.splits["test"].num_examples // batch_size

    train_steps = 4 * 100000
    epochs = train_steps // train_steps_per_epoch

    print(f'Number of epochs = {epochs}')

    if os.path.exists(checkpoint_path) and args.resume_training:
        model = tf.keras.saving.load_model(checkpoint_path)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1
    )
