"""meter_values_dataset_stage1 dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import sys
sys.path.append('..')

import process_labels_label_studio


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for meter_values_dataset_stage1 dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Some description here
  """

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(meter_values_dataset_stage1): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(1024, 1024, 3), dtype=tf.uint8),
            'image/filename': tfds.features.Text(),
            'image/id': tf.int64,
            'objects': tfds.features.Sequence({
                'area': tf.int64,
                'bbox': tfds.features.Tensor(shape=(12,), dtype=tf.float32),#tfds.features.BBoxFeature(),
                'id': tf.int64,
                'is_crowd': tf.bool,
                'label': tfds.features.ClassLabel(num_classes=4),
            }),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        #supervised_keys=('image', 'image/filename', 'image/id', 'objects'),  # Set to `None` to disable
        homepage=None,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    # TODO(meter_values_dataset): Downloads the data and defines the splits
    # data_path is a pathlib-like `Path('<manual_dir>/data.zip')`
    path = dl_manager.manual_dir #/ 'data.zip'
    # Extract the manually downloaded `data.zip`
    #path = dl_manager.extract(archive_path)

    # TODO(MeterValuesDataset): Returns the Dict[split names, Iterator[Key, Example]]
    self.width = 1024
    self.height = 1024
    partition_train = 0.8
    partition_val = 0.1
    partition_test = 0.1
    images_info = process_labels_label_studio.get_images_info(path / 'labels.json')
    max_samples_train = np.floor(len(images_info) * partition_train)
    max_samples_val = np.floor(len(images_info) * partition_val)
    max_samples_test = np.floor(len(images_info) * partition_test)
    self.iter = process_labels_label_studio.generate_examples_stage1(
      images_info, path, self.width, self.height)
    return {
        'train': self._generate_examples(max_samples_train),
        'validation': self._generate_examples(max_samples_val),
        'test': self._generate_examples(max_samples_test),
    }

  def _generate_examples(self, max_samples):
    """Yields examples."""
    # TODO(meter_values_dataset_stage1): Yields (key, example) tuples from the dataset
    str2int = {
      'analog': 0,
      'digital': 1,
      'analog_illegible': 2,
      'digital_illegible': 3
    }

    index = 0
    while index < max_samples:
      
      index += 1

      im_resized, label, bbox, keypoints, image_filename = next(self.iter)
      
      final_bbox = np.array([
        bbox[1] / self.height,
        bbox[0] / self.width,
        (bbox[1] + bbox[3]) / self.height,
        (bbox[0] + bbox[2]) / self.width,
        keypoints[0][1] / self.height,
        keypoints[0][0] / self.width,
        keypoints[1][1] / self.height,
        keypoints[1][0] / self.width,
        keypoints[2][1] / self.height,
        keypoints[2][0] / self.width,
        keypoints[3][1] / self.height,
        keypoints[3][0] / self.width,
      ], dtype=np.float32)

      yield index, {
          'image': im_resized.astype(np.uint8),
          'image/filename': image_filename,
          'image/id': index,
          'objects': {
            'area': [bbox[2] * bbox[3]],
            'bbox': [final_bbox],#[bbox_feature],
            'id': [0],
            'is_crowd': [False],
            'label': [str2int[label[0].lower()]]
          }
      }
