import typing as tp

import logging

import tensorflow as tf


def set_physical_gpu_memory_limit(memory_limit: int):
    """
    Sets the memory limit for all GPUs.
    
    Args:
        memory_limit: the memory limit for each gpu
    
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return
    logical_gpus = tf.config.list_logical_devices('GPU')
    logging.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    for gpu_idx, gpu in enumerate(gpus):
        try:
            tf.config.set_logical_device_configuration(gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            logging.error(f'GPU[{gpu_idx}]: {e}')
