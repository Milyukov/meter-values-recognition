import pprint
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from six import BytesIO
from urllib.request import urlopen

from official.vision.ops.preprocess_ops import resize_and_crop_image
from official.vision.utils.object_detection import visualization_utils
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder

pp = pprint.PrettyPrinter(indent=4) # Set Pretty Print Indentation
print(tf.__version__) # Check the version of tensorflow used

test_data_input_path = './counters/test-00000-of-00001.tfrecord'
export_dir ='./exported_model_counters/'

HEIGHT, WIDTH = 512, 512

category_index={
    0: {
            "id": 0,
            "name": "+"
        },
    1:  {
            "id": 1,
            "name": "-"
        },
    2:  {
            "id": 2,
            "name": "0"
        },
    3:  {
            "id": 3,
            "name": "1"
        },
    4:  {
            "id": 4,
            "name": "2"
        },
    5:  {
            "id": 5,
            "name": "3"
        },
    6:  {
            "id": 6,
            "name": "4"
        },
    7:  {
            "id": 7,
            "name": "5"
        },
    8:  {
            "id": 8,
            "name": "6"
        },
    9:  {
            "id": 9,
            "name": "7"
        },
    10: {
            "id": 10,
            "name": "8"
        },
    11: {
            "id": 11,
            "name": "9"
        },
    12: {
            "id": 12,
            "name": "COLON"
        },
    13: {
            "id": 13,
            "name": "FLOATP"
        },
    14: {
            "id": 14,
            "name": "M"
        },
    15: {
            "id": 15,
            "name": "Q"
        },
    16: {
            "id": 16,
            "name": "R"
        },
    17: {
            "id": 17,
            "name": "T"
        },
    18: {
            "id": 18,
            "name": "U"
        },
    19: {
            "id": 19,
            "name": "V"
        },
    20: {
            "id": 20,
            "name": "_"
        },
    21: {
            "id": 21,
            "name": "arrow"
        },
    22: {
            "id": 22,
            "name": "Аналоговый счётчик"
        },
    23: {
            "id": 23,
            "name": "С"
        },
    24: {
            "id": 24,
            "name": "Цифровой счётчик"
        },
    25: {
            "id": 25,
            "name": "Ч"
        }
}
tf_ex_decoder = TfExampleDecoder()

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  image = None
  if(path.startswith('http')):
    response = urlopen(path)
    image_data = response.read()
    image_data = BytesIO(image_data)
    image = Image.open(image_data)
  else:
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))

  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (1, im_height, im_width, 3)).astype(np.uint8)



def build_inputs_for_object_detection(image, input_image_size):
  """Builds Object Detection model inputs for serving."""
  image, _ = resize_and_crop_image(
      image,
      input_image_size,
      padded_size=input_image_size,
      aug_scale_min=1.0,
      aug_scale_max=1.0)
  return image

def show_batch(raw_records, num_of_examples):
  plt.figure(figsize=(20, 20))
  use_normalized_coordinates=True
  min_score_thresh = 0.5
  for i, serialized_example in enumerate(raw_records):
    if i == num_of_examples:
      break
    plt.subplot(1, 3, i + 1)
    decoded_tensors = tf_ex_decoder.decode(serialized_example)
    image = decoded_tensors['image'].numpy().astype('uint8')
    scores = np.ones(shape=(len(decoded_tensors['groundtruth_boxes'])))
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image,
        decoded_tensors['groundtruth_boxes'].numpy(),
        decoded_tensors['groundtruth_classes'].numpy().astype('int'),
        scores,
        category_index=category_index,
        use_normalized_coordinates=use_normalized_coordinates,
        max_boxes_to_draw=200,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        instance_masks=None,
        line_thickness=4)

    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Image-{i+1}')
  plt.show()

num_of_examples = 3

test_ds = tf.data.TFRecordDataset(test_data_input_path)#.take(num_of_examples)
matplotlib.use('TKAgg', force=True)
print("Using:",matplotlib.get_backend())
show_batch(test_ds, num_of_examples)

imported = tf.saved_model.load(export_dir)
model_fn = imported.signatures['serving_default']

input_image_size = (HEIGHT, WIDTH)
plt.figure(figsize=(20, 20))
min_score_thresh = 0.50 # Change minimum score for threshold to see all bounding boxes confidences.

for i, serialized_example in enumerate(test_ds):
  #plt.subplot(1, 3, i+1)
  decoded_tensors = tf_ex_decoder.decode(serialized_example)
  image = build_inputs_for_object_detection(decoded_tensors['image'], input_image_size)
  image = tf.expand_dims(image, axis=0)
  image = tf.cast(image, dtype = tf.uint8)
  image_np = image[0].numpy()
  result = model_fn(image)
  visualization_utils.visualize_boxes_and_labels_on_image_array(
      image_np,
      result['detection_boxes'][0].numpy(),
      result['detection_classes'][0].numpy().astype(int),
      result['detection_scores'][0].numpy(),
      category_index=category_index,
      use_normalized_coordinates=False,
      max_boxes_to_draw=200,
      min_score_thresh=min_score_thresh,
      agnostic_mode=False,
      instance_masks=None,
      line_thickness=4)
  plt.imshow(image_np)
  plt.axis('off')

  plt.show()