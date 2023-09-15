# for each spatial location
# for each scale of FPN
# classification head outputs KA-size vector of values between 0 and 1 (sigmoid applied to each value)
# from GT if anchor is assigned
# take corresponding class and create one-hot vector
#  000 ......................... 001000......0000
# |_K values_1|_K values_2|...|_K_Values_i_|....



# for given GT example
# of class k
# on each scale
# find corresponding anchors
# for each anchor a with location x, y
# compute log loss for classification head
# -log(cls[x, y, a * K + k])
# compute L1 error for regression head
# L1(reg[x, y], gt_ref[x, y]), where gt_reg is all zeros except [a*12:a*12 + 12] 
# where we have offset from anchor box to nearby GT object
# 

import tensorflow as tf
import tensorflow_models as tfm

example_path = '/home/gleb/tensorflow_datasets/MeterValuesDataset/1.0.0/MeterValuesDataset-train.tfrecord-00000-of-00001'

# height = anchor_size / sqrt(aspect_ratio)
# width = anchor_size * sqrt(aspect_ratio)
# aspect ratio is a ratio of anchor width to anchor height

anchor_sizes = [32, 64, 128, 256, 512]
scales = [[1, 2 ** (1 /3), 2 ** (2 / 3)]] * len(anchor_sizes)
aspect_ratios = [[0.5, 1.0, 1.5]] * len(anchor_sizes)
strides = [1] * len(anchor_sizes)

gen = tfm.vision.anchor_generator.AnchorGenerator(
    anchor_sizes, scales, aspect_ratios, strides, clip_boxes=True
)

def decode_fn(record_bytes):
  return tf.io.parse_single_example(
      # Data
      record_bytes,

      # Schema
      {"image": tf.io.FixedLenFeature([], dtype=tf.string),
       "label": tf.io.FixedLenFeature([], dtype=tf.int64),
       "bbox": tf.io.FixedLenFeature([1, 1, 12], dtype=tf.float32)}
  )

for batch in tf.data.TFRecordDataset([example_path]).map(decode_fn).batch(2):
    # read batch from dataset
    images = []
    for im in batch["image"]:
        images.append(tf.io.decode_jpeg(im))
    labels = batch["label"]
    bboxes = batch["bbox"]
    
    # infer model on images
    # get predictions
    # find corresponding anchor boxes
    for index, anchor in enumerate(gen(image_size=(100, 100))):
        # compute IoU
        
        # if IoU > 0.7
        # save candidate
        pass
    
    # compute loss for each head

