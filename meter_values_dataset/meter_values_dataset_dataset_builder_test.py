"""meter_values_dataset dataset."""

from . import meter_values_dataset_dataset_builder
import tensorflow_datasets as tfds

class MeterValuesDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for meter_values_dataset dataset."""
  # TODO(meter_values_dataset):
  DATASET_CLASS = meter_values_dataset_dataset_builder.Builder
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
