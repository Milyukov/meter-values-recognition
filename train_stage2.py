from datetime import datetime
import os
import argparse
import tensorflow_datasets as tfds

from model.data_processing_stage2 import LabelEncoder, preprocess_data
from model.model_stage2 import *
import tensorflow as tf

import gc
import psutil

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # change this value as needed
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

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", help="path to dataset", type=str, 
                    default='/mnt/images/counters-datasets/meter_values_dataset_stage2')
parser.add_argument("--resume_training", help="Boolean flag wheather to resume training if it was interrupted", 
                    action="store_true")

if __name__ == '__main__':
    model_dir = "retinanet/"
    label_encoder = LabelEncoder()

    num_classes = 17
    batch_size = 16

    learning_rates = [0.00001, 0.000005]#[2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [250] #[125, 250,500, 240000, 360000] 
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, resnet50_backbone)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    model.compile(loss=loss_fn, optimizer=optimizer)#, run_eagerly=True)

    logdir = "logs/stage2/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    checkpoint_path = "retinanet/stage2.keras"
    checkpoint_backup_path = "retinanet/stage2_bak.keras"
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="loss",
            save_best_only=False,
            save_weights_only=False,
            save_freq=200,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_backup_path,
            monitor="loss",
            save_best_only=False,
            save_weights_only=False,
            save_freq=250,
            verbose=1,
        ),
        MemoryUsageCallbackExtended(),
        tensorboard_callback
    ]

    (train_dataset, val_dataset), dataset_info = tfds.load(
        "meter_values_dataset_stage2", split=["train", "validation"], with_info=True,
        read_config=tfds.ReadConfig(try_autocache=False), data_dir=parser.dataset_path
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
    train_dataset = train_dataset.prefetch(0)

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
        dataset_info.splits["validation"].num_examples // batch_size

    train_steps = 4 * 100000
    epochs = train_steps // train_steps_per_epoch

    if os.path.exists(checkpoint_path) and parser.resume_training:
        model = tf.keras.saving.load_model(checkpoint_path)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )
