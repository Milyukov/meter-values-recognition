"""meter_values_dataset_stage1 dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

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
    archive_path = dl_manager.manual_dir / 'data.zip'
    # Extract the manually downloaded `data.zip`
    path = dl_manager.extract(archive_path)

    # TODO(MeterValuesDataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path, 'train'),
        'test': self._generate_examples(path, 'test'),
    }

  def _generate_examples(self, path, dataset_name):
    """Yields examples."""
    # TODO(meter_values_dataset_stage1): Yields (key, example) tuples from the dataset
    partition = 0.9
    str2int = {
      'analog': 0,
      'digital': 1,
      'analog_illegible': 2,
      'digital_illegible': 3
    }

    index = 0
    width = 1024
    height = 1024
    images_info = process_labels_label_studio.get_images_info(path / 'labels.json')
    max_samples = np.floor(len(images_info) * partition)
    for im_resized, label, bbox, keypoints in process_labels_label_studio.generate_examples(
      images_info, path, width, height):

      index += 1

      if dataset_name == 'train' and index > max_samples:
        break
      elif dataset_name == 'test' and index <= max_samples:
        continue
      
      # bbox_feature = tfds.features.BBox(
      #   ymin=bbox[1] / height,
      #   xmin=bbox[0] / width,
      #   ymax=(bbox[1] + bbox[3]) / height,
      #   xmax=(bbox[0] + bbox[2]) / width)
      
      final_bbox = np.array([
        bbox[1] / height,
        bbox[0] / width,
        (bbox[1] + bbox[3]) / height,
        (bbox[0] + bbox[2]) / width,
        keypoints[0][1] / height,
        keypoints[0][0] / width,
        keypoints[1][1] / height,
        keypoints[1][0] / width,
        keypoints[2][1] / height,
        keypoints[2][0] / width,
        keypoints[3][1] / height,
        keypoints[3][0] / width,
      ], dtype=np.float32)

      yield index, {
          'image': im_resized.astype(np.uint8),
          'image/filename': '',
          'image/id': index,
          'objects': {
            'area': [bbox[2] * bbox[3]],
            'bbox': [final_bbox],#[bbox_feature],
            'id': [0],
            'is_crowd': [False],
            'label': [str2int[label[0].lower()]]
          }
      }
