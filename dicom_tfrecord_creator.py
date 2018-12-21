# 
# refer to "build_imagenet_data.py" in tensorflow 
#==========================================================================

"""
Read CT DICOM Files and generate tfrecords files. 
Input:  data_dir

    In the data_dir, dicom file should be organized with ImageNet rules.

    data_dir-
            |
            class 1-
                    |
                    file1.dy
                    filr2.dy
            class 2-
                    | 
                    file21.dy            

Output: tfrecords list

tfrecords 
    keys_to_features = {
        "image_data": tf.FixedLenFeature((), tf.int32, default_value=tf.zeros([]), dtype=tf.int32),
        "label": tf.FixedLenFeature((), tf.int32, default_value=0
                                    , dtype=tf.int32)),
    }

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import six
import tensorflow as tf

import pydicom

tf.app.flags.DEFINE_string('train_directory', '/data/train',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/data/validation',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/output/',
                           'Output data directory')
tf.app.flags.DEFINE_integer('train_shards', 128,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 8,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_bool('object_detection', False, 'Whether include object detection by adding box information')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   n01440764
#   n01443537
#   n01484850
# where each line corresponds to a label expressed as string(synset in imagenet). We map
# each synset contained in the file to an integer (based on the alphabetical
# ordering). See below for details.
tf.app.flags.DEFINE_string('labels_file',
                           '/data/medical_all_labels.txt',
                           'Medical Labels file')

# This file containing mapping from synset to human-readable label.
# Assumes each line of the file looks like:
#
#   n02119247    sickness 1
#   n02119359    sickness 2
#   n02119477    sickness 3
#
# where each line corresponds to a unique mapping. Note that each line is
# formatted as <synset>\t<human readable label>.
tf.app.flags.DEFINE_string('medical_label2name_file',
                           '/data/medical_labels2names.txt',
                           'Medical labels to names mapping file')

# This file is the output of process_bounding_box.py
# Assumes each line of the file looks like:
#
#   n00007846_64193.dicom,0.0060,0.2620,0.7545,0.9940
#
# where each line corresponds to one bounding box annotation associated
# with an dicom file. Each line can be parsed as:
#
#   <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>
#
# Note that there might exist mulitple bounding box annotations associated
# with an dicom file.
tf.app.flags.DEFINE_string('bounding_box_file',
                           '/data/medical_bounding_boxes.csv',
                           'Medical Bounding box file')

FLAGS = tf.app.flags.FLAGS

#TODO: operation on compressed dicom file
class DicomCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    self.image_data = None


  def decode_dicom(self, dicom_data):
    image_data = None  
    return image_data

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _int64_feature_ndarray(value):
  """Wrapper for inserting int64 features into Example proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if six.PY3 and isinstance(value, six.text_type):           
    value = six.binary_type(value, encoding='utf-8') 
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, pixel_data, label, synset, human, bbox,
                        height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
    bbox: list of bounding boxes; each box is a list of integers
      specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong to
      the same label as the image label.
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  xmin = []
  ymin = []
  xmax = []
  ymax = []

  if bbox is not None:    
    for b in bbox:
      assert len(b) == 4
      # pylint: disable=expression-not-assigned
      [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
      # pylint: enable=expression-not-assigned

  colorspace = 'CT'
  channels = 1
  file_format = 'DICOM'

  #Important, reshape list-like ndarray
  pixel_data_reshape = pixel_data.reshape(-1)
  #list_pixel_data = pixel_data.tolist()

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/synset': _bytes_feature(synset),
      'image/class/text': _bytes_feature(human),
      'image/object/bbox/xmin': _float_feature(xmin),
      'image/object/bbox/xmax': _float_feature(xmax),
      'image/object/bbox/ymin': _float_feature(ymin),
      'image/object/bbox/ymax': _float_feature(ymax),
      'image/object/bbox/label': _int64_feature([label] * len(xmin)),
      'image/format': _bytes_feature(file_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/pixel': _int64_feature_ndarray(pixel_data_reshape)}))
  return example


def _process_dicom(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.dcm'.
    coder: instance of DicomCoder to provide dicom coding utils.
  Returns:
    image_raw_data: maxtrix of uint32, .
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """

  #filename = get_testdata_files(filename)[0]
  print(filename)
  ds = pydicom.dcmread(filename)    
  
  hu_image_raw_data = ds.pixel_array

  # TODO: 
  # image_raw_data = coder(image_raw_data)
  height = hu_image_raw_data.shape[0]
  width = hu_image_raw_data.shape[1]
  # DCM file may not have these two tags.
  #window_center = ds.WindowCenter
  #window_width = ds.WindowWidth

  return hu_image_raw_data, height, width


def _process_dicom_files_batch(coder, thread_index, ranges, name, filenames,
                               synsets, labels, humans, bboxes, num_shards):
  """Processes and saves list of dicom as TFRecord in 1 thread.

  Args:
    coder: instance of DicomCoder to provide TensorFlow dicom coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an dicom file
    synsets: list of strings; each string is a unique WordNet ID
    labels: list of integer; each integer identifies the ground truth
    humans: list of strings; each string is a human-readable label
    bboxes: list of bounding boxes for each dicom. Note that each entry in this
      list might contain from 0+ entries corresponding to the number of bounding
      box annotations for the dicom.
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d.tfrecords' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      synset = synsets[i]
      human = humans[i]

      if bboxes is not None:
        bbox = bboxes[i]
      else:
        bbox = None

      image_buffer, height, width = _process_dicom(filename, coder)

      example = _convert_to_example(filename, image_buffer, label,
                                    synset, human, bbox,
                                    height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()



# TODO: process dicom
def _process_dicom_files(name, filenames, synsets, labels, humans,
                         bboxes, num_shards):
  """Process and save list of dicom files as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an dicom file
    synsets: list of strings; each string is a unique WordNet ID
    labels: list of integer; each integer identifies the ground truth
    humans: list of strings; each string is a human-readable label
    bboxes: list of bounding boxes for each dicom. Note that each entry in this
      list might contain from 0+ entries corresponding to the number of bounding
      box annotations for the dicom.
    num_shards: integer number of shards for this data set.
  """
  assert len(filenames) == len(synsets)
  assert len(filenames) == len(labels)
  assert len(filenames) == len(humans)
  #assert len(filenames) == len(bboxes)

  # Break all images/dicoms into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = DicomCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames,
            synsets, labels, humans, bboxes, num_shards)
    t = threading.Thread(target=_process_dicom_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_dicom_files(data_dir, labels_file):
  """Build a list of all images/dicom files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of images/dicom.

      Assumes that the images/dicom data set resides in JPEG files located in
      the following directory structure.

        data_dir/n01440764/ILSVRC2012_val_00000293.JPEG/dicom
        data_dir/n01440764/ILSVRC2012_val_00000543.JPEG/dicom

      where 'n01440764' is the unique synset label associated with these images.

    labels_file: string, path to the labels file.

      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        n01440764
        n01443537
        n01484850
      where each line corresponds to a label expressed as a synset. We map
      each synset contained in the file to an integer (based on the alphabetical
      ordering) starting with the integer 1 corresponding to the synset
      contained in the first line.

      The reason we start the integer labels at 1 is to reserve label 0 as an
      unused background class.

  Returns:
    filenames: list of strings; each string is a path to an image/dicom file.
    synsets: list of strings; each string is a unique WordNet ID.
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % data_dir)
  challenge_synsets = [l.strip() for l in
                       tf.gfile.FastGFile(labels_file, 'r').readlines()]

  labels = []
  filenames = []
  synsets = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of JPEG files and labels.
  # TODO: 
  for synset in challenge_synsets:
    #dicom_file_path = '%s/%s/*.dcm-+' % (data_dir, synset)
    dicom_file_path = '%s/%s/*.*' % (data_dir, synset)
    matching_files = tf.gfile.Glob(dicom_file_path)

    labels.extend([label_index] * len(matching_files))
    synsets.extend([synset] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(challenge_synsets)))
    label_index += 1

  # Shuffle the ordering of all image/dicom files in order to guarantee
  # random ordering of the images/dicom with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  synsets = [synsets[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d DICOM format files across %d labels inside %s.' %
        (len(filenames), len(challenge_synsets), data_dir))
  return filenames, synsets, labels

def _find_human_readable_labels(synsets, synset_to_human):
  """Build a list of human-readable labels.

  Args:
    synsets: list of strings; each string is a unique WordNet ID.
    synset_to_human: dict of synset to human labels, e.g.,
      'n02119022' --> 'red fox, Vulpes vulpes'

  Returns:
    List of human-readable strings corresponding to each synset.
  """
  humans = []
  for s in synsets:
    assert s in synset_to_human, ('Failed to find: %s' % s)
    humans.append(synset_to_human[s])
  return humans


def _find_dicom_bounding_boxes(filenames, dicom_to_bboxes):
  """Find the bounding boxes for a given image file.

  Args:
    filenames: list of strings; each string is a path to an image file.
    dicom_to_bboxes: dictionary mapping dicom/image file names to a list of
      bounding boxes. This list contains 0+ bounding boxes.
  Returns:
    List of bounding boxes for each image. Note that each entry in this
    list might contain from 0+ entries corresponding to the number of bounding
    box annotations for the image.
  """
  num_dicom_bbox = 0
  bboxes = []
  for f in filenames:
    basename = os.path.basename(f)
    if basename in dicom_to_bboxes:
      bboxes.append(dicom_to_bboxes[basename])
      num_dicom_bbox += 1
    else:
      bboxes.append([])
  print('Found %d images with bboxes out of %d images' % (
      num_dicom_bbox, len(filenames)))
  return bboxes


def _process_dataset(name, directory, num_shards, synset_to_human,
                     dicom_to_bboxes):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    synset_to_human: dict of synset to human labels, e.g.,
      'n02119022' --> 'sickness 1'
    dicom_to_bboxes: dictionary mapping dicom file names to a list of
      bounding boxes. This list contains 0+ bounding boxes.
  """

  # TODO:
  filenames, synsets, labels = _find_dicom_files(directory, FLAGS.labels_file)
  humans = _find_human_readable_labels(synsets, synset_to_human)
  if FLAGS.object_detection:
    bboxes = _find_dicom_bounding_boxes(filenames, dicom_to_bboxes)
  else:
    bboxes = None
  #TODO: 
  _process_dicom_files(name, filenames, synsets, labels,
                       humans, bboxes, num_shards)


def _build_synset_lookup(medical_label2name_file):
  """Build lookup for synset to human-readable label.

  Args:
    medical_label2name_file: string, path to file containing mapping from
      synset to human-readable label.

      Assumes each line of the file looks like:

        n02119247    black fox
        n02119359    silver fox
        n02119477    red fox, Vulpes fulva

      where each line corresponds to a unique mapping. Note that each line is
      formatted as <synset>\t<human readable label>.

  Returns:
    Dictionary of synset to human labels, such as:
      'n02119022' --> 'red fox, Vulpes vulpes'
  """
  lines = tf.gfile.FastGFile(medical_label2name_file, 'r').readlines()
  synset_to_human = {}
  for l in lines:
    if l:
      parts = l.strip().split('\t')
      assert len(parts) == 2
      synset = parts[0]
      human = parts[1]
      synset_to_human[synset] = human
  return synset_to_human


def _build_bounding_box_lookup(bounding_box_file):
  """Build a lookup from dicom/image file to bounding boxes.

  Args:
    bounding_box_file: string, path to file with bounding boxes annotations.

      Assumes each line of the file looks like:

        n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940

      where each line corresponds to one bounding box annotation associated
      with an image/dicom. Each line can be parsed as:

        <dicom file name>, <xmin>, <ymin>, <xmax>, <ymax>

      Note that there might exist mulitple bounding box annotations associated
      with an dicom/image file. This file is the output of process_bounding_boxes.py.

  Returns:
    Dictionary mapping dicom/image file names to a list of bounding boxes. This list
    contains 0+ bounding boxes.
  """
  lines = tf.gfile.FastGFile(bounding_box_file, 'r').readlines()
  images_to_bboxes = {}
  num_bbox = 0
  num_image = 0
  for l in lines:
    if l:
      parts = l.split(',')
      assert len(parts) == 5, ('Failed to parse: %s' % l)
      filename = parts[0]
      xmin = float(parts[1])
      ymin = float(parts[2])
      xmax = float(parts[3])
      ymax = float(parts[4])
      box = [xmin, ymin, xmax, ymax]

      if filename not in images_to_bboxes:
        images_to_bboxes[filename] = []
        num_image += 1
      images_to_bboxes[filename].append(box)
      num_bbox += 1

  print('Successfully read %d bounding boxes '
        'across %d images.' % (num_bbox, num_image))
  return images_to_bboxes


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  # Build a map from synset to human-readable label.
  synset_to_human = _build_synset_lookup(FLAGS.medical_label2name_file)

  if FLAGS.object_detection:
    dicom_to_bboxes = _build_bounding_box_lookup(FLAGS.bounding_box_file)
  else:
    dicom_to_bboxes = None

  # Run it!
  _process_dataset('validation', FLAGS.validation_directory,
                   FLAGS.validation_shards, synset_to_human, dicom_to_bboxes)
  _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards,
                   synset_to_human, dicom_to_bboxes)


if __name__ == '__main__':
  tf.app.run()