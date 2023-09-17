import tensorflow_datasets as tfds

from data_processing import LabelEncoder, preprocess_data
from model import *

import datetime
import multiprocessing
from tensorboard.plugins.hparams import api as hp

import tensorflow as tf

# tune HP
def run(queue, run_dir, hparams):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8 # change this value as needed
    session = tf.compat.v1.Session(config=config)

    label_encoder = LabelEncoder()
    batch_size = 1
    train_dataset, dataset_info = tfds.load(
        "meter_values_dataset", split=["train"], with_info=True, data_dir="/home/gleb/tensorflow_datasets",
        read_config=tfds.ReadConfig(try_autocache=False)
    )
    train_dataset = train_dataset[0]
    train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=1)#autotune)
    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    train_dataset = train_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=1#autotune
    )
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    
    num_classes = 4
    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, resnet50_backbone)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'])
    model.compile(loss=loss_fn, optimizer=optimizer)#, run_eagerly=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_dir, histogram_freq=1)
    terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()
    
    epochs = 30
    with tf.summary.create_file_writer(run_dir).as_default():
        model.fit(
            train_dataset.take(1),
            epochs=epochs,
            callbacks=[tensorboard_callback, terminate_on_nan],
            verbose=1,
        )
        loss = model.evaluate(train_dataset.take(1))
        queue.put(loss)
        tf.summary.scalar('Loss', loss, step=1)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    tf.test.gpu_device_name()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(f"tensorflow version: {tf.__version__}")

    log_dir = "logs/hparam_tuning/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    HP_LR = hp.HParam(
        'learning_rate', 
        hp.Discrete([0.1 ** i for i in range(6)])
        )

    session_num = 0
    for hp_lr in HP_LR.domain.values:
        tf.keras.backend.clear_session()

        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"Mem before: {tf.config.experimental.get_memory_usage('GPU:0')}")

        hparams = {
            "learning_rate": hp_lr
        }

        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)

        queue = multiprocessing.Queue()
        p1 = multiprocessing.Process(target=run, args=(queue, log_dir + '/' + run_name, hparams))
        p1.start()
        p1.join()
        results = queue.get()
        p1.close()

        session_num += 1
