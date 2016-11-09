#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2016, MBARI"
__license__ = "GNU License"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__status__ = "Development"
__doc__ = '''

This script runs learning algorithms on the AESA training data set

Based on the TensorFlow code:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py

Prerequisites:

@undocumented: __doc__ parser
@author: __author__
@status: __status__
@license: __license__
'''

import json
import conf
import sys
import argparse
import os
import util_plot
import numpy as np
import util
import time
import transfer_model as transfer_model
import tensorflow as tf
from tensorflow.python.platform import gfile
from scipy.misc import imresize


def process_command_line():
    from argparse import RawTextHelpFormatter

    examples = 'Examples:' + '\n\n'
    examples += sys.argv[0] + " --image_dir /tmp/data/images_by_group/cropped_images/" \
                              " --bottleneck_dir /tmp/data/images_by_group/cropped_images/bottleneck" \
                              " --model_dir /tmp/model_output/default"
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description='Process LTSA data for last 3 days',
                                     epilog=examples)

    # Input and output file flags.
    parser.add_argument('--image_dir', type=str, required=True,  help="Path to folders of labeled images.")
    # where the model information lives
    parser.add_argument('--model_dir', type=str, default=os.path.join( "/tmp/tfmodels/img_classify", str(int(time.time()))), help='Directory for storing model info')
 
    # Details of the training configuration.
    parser.add_argument('--num_steps', type=int, default=15000, help="How many training steps to run before ending.")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="How large a learning rate to use when training.")
    parser.add_argument('--testing_percentage', type=int, default=10, help="What percentage of images to use as a test set.")
    parser.add_argument('--validation_percentage', type=int, default=10, help="What percentage of images to use as a validation set.")
    parser.add_argument('--eval_step_interval', type=int, default=10, help="How often to evaluate the training results.")
    parser.add_argument('--train_batch_size', type=int, default=100, help="How many images to train on at a time.")
    parser.add_argument('--test_batch_size', type=int, default=500,
                        help="""How many images to test on at a time. This
                        test set is only used infrequently to verify
                        the overall accuracy of the model.""")
    parser.add_argument( '--validation_batch_size', type=int, default=100,
                        help="""How many images to use in an evaluation batch. This validation
                        set is used much more often than the test set, and is an early
                        indicator of how accurate the model is during training.""")

    # File-system cache locations.
    parser.add_argument('--incp_model_dir', type=str, default='/tmp/imagenet', help="""Path to graph.pb for a given model""")
    parser.add_argument('--bottleneck_dir', type=str, default='/tmp/bottlenecks', help="Path to cache bottleneck layer values as files.")
    parser.add_argument('--final_tensor_name', type=str, default='final_result', help="The name of the output classification layer in the retrained graph.")

    # Controls the distortions used during training.
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--flip_left_right', action='store_true', default=False, help="Whether to randomly flip the training images horizontally.")
    group.add_argument('--rotate90', action='store_true', default=False, help="Whether to randomly rotate the training images 90.")
    parser.add_argument('--random_crop', type=int, default=0, help="""A percentage determining how much of a margin to randomly crop off the training images.""")
    parser.add_argument('--random_scale', type=int, default=0, help="""A percentage determining how much to randomly scale up the size of the training images by.""")
    parser.add_argument('--random_brightness', type=int, default=0, help="""A percentage determining how much to randomly multiply the training image input pixels up or down by.""")

    # Custom selections AESA training set
    parser.add_argument('--exclude_unknown', dest='exclude_unknown', action='store_true', help="Exclude classes that include the unknown category")
    parser.add_argument('--metrics_plot_name', type=str, default='metrics_plot.png', help="""The name of the metric plot""")

    args = parser.parse_args()
    return args

def create_inception_graph(model_filename):
  """"Creates a graph from saved GraphDef file and returns a Graph object.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Session() as sess:

    # import the graph and give me nodes where we want to pull the bottleneck data from
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              conf.BOTTLENECK_TENSOR_NAME, conf.JPEG_DATA_TENSOR_NAME,
              conf.RESIZED_INPUT_TENSOR_NAME]))

  return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             bottleneck_tensor):
  """Retrieves or calculates bottleneck values for an image.

  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be modulo-ed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string  of the subfolders containing the training
    images.
    category: Name string of which  set to pull images from: training, testing,
    or validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: The tensor to feed loaded jpeg data into.
    bottleneck_tensor: The output tensor for the bottleneck values.

  Returns:
    Numpy array of values produced by the bottleneck layer for the image.
    Original image path string
  """
  label_lists = image_lists[label_name]
  sub_dir = label_lists['dir']
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  util.ensure_dir(sub_dir_path)
  bottleneck_path = util.get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category)
  image_path = util.get_image_path(image_lists, label_name, index, image_dir, category)
  if not os.path.exists(bottleneck_path):
    print('Creating bottleneck at ' + bottleneck_path)
    if not gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    bottleneck_values = util.run_bottleneck_on_image(sess, image_data,
                                                jpeg_data_tensor,
                                                bottleneck_tensor)
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
      bottleneck_file.write(bottleneck_string)

  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()

  bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values, image_path


def add_images(sess, paths, model_dir):

  filename_queue = tf.train.string_input_producer(paths)
  reader = tf.WholeFileReader()
  # Read a whole file from the queue, the first returned value in the tuple is the
  # filename which we are ignoring.
  _, image_file = reader.read(filename_queue)

  # Decode the image as a JPEG file, this will turn it into a Tensor which we can
  # then use in training.
  nth_image = 10
  num_images = int(len(paths)/nth_image)
  image = tf.image.decode_jpeg(image_file)
  image_tensors = np.zeros((num_images, conf.MODEL_INPUT_WIDTH, conf.MODEL_INPUT_WIDTH, 3), dtype=np.float32)

  # Add an Op to initialize all variables.
  init_op = tf.initialize_all_variables()

  with sess.as_default():

    # Run the Op that initializes all variables.
    sess.run(init_op)

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    j = 0

    # Write summary
    writer = tf.train.SummaryWriter(model_dir)

    if num_images > 0:
      for count, name in enumerate(paths, 1):
        if count % nth_image == 0:
          image_tensor = image.eval()
          image_tensors[j] = imresize(image_tensor, [conf.MODEL_INPUT_WIDTH,conf.MODEL_INPUT_WIDTH])
          print(str(j) + ' images files created.')
          j += 1

      # Add image summary
      summary_op = tf.image_summary("plot", image_tensors, num_images )
      summary = sess.run(summary_op)
      writer.add_summary(summary)

    writer.close()

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    args = process_command_line()

    print("Using model directory {0} and model from {1}".format(args.model_dir, conf.DATA_URL))
    # Set up the pre-trained graph.
    util.maybe_download_and_extract(data_url=conf.DATA_URL, dest_dir=args.incp_model_dir)
    model_filename = os.path.join(args.incp_model_dir, conf.MODEL_GRAPH_NAME)
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor=(create_inception_graph(model_filename))

    labels_list = None
    output_labels_file = os.path.join(args.model_dir, "output_labels.json")

    # Look at the folder structure, and create lists of all the images.
    image_lists = util.create_image_lists(args.exclude_unknown, output_labels_file,
        args.image_dir, args.testing_percentage,
        args.validation_percentage)
    class_count = len(image_lists.keys())
    if class_count == 0:
      print('No valid folders of images found at ' + args.image_dir)
      exit(-1)
    if class_count == 1:
      print('Only one valid folder of images found at ' + args.image_dir +
            ' - multiple classes are needed for classification.')
      exit(-1)

    # See if the command-line flags mean we're applying any distortions.
    do_distort_images =  (args.flip_left_right or args.rotate90
                          or (args.random_crop != 0) or (args.random_scale != 0) or
                          (args.random_brightness != 0))
    sess = tf.Session()

    if do_distort_images:
      # We will be applying distortions, so setup the operations we'll need.
      distorted_jpeg_data_tensor, distorted_image_tensor = util.add_input_distortions(
          args.rotate90, args.flip_left_right, args.random_crop, args.random_scale,
          args.random_brightness)
    else:
      # We'll make sure we've calculated the 'bottleneck' image summaries and
      # cached them on disk.
      util.cache_bottlenecks(sess, image_lists, args.image_dir, args.bottleneck_dir,
                        jpeg_data_tensor, bottleneck_tensor)
    # Define the custom estimator
    model_fn = transfer_model.make_model_fn(class_count, args.final_tensor_name, args.learning_rate)
    model_params = {}
    classifier = tf.contrib.learn.Estimator(
        model_fn=model_fn, params=model_params, model_dir=args.model_dir)

    train_bottlenecks, train_ground_truth = util.get_all_cached_bottlenecks(sess, image_lists, 'training', args.bottleneck_dir, args.image_dir, jpeg_data_tensor, bottleneck_tensor)
    train_bottlenecks = np.array(train_bottlenecks)
    train_ground_truth = np.array(train_ground_truth)

    # then run the training
    print("Starting training for %s steps max" % args.num_steps)
    classifier.fit(
        x=train_bottlenecks.astype(np.float32),
        y=train_ground_truth, batch_size=50,
        max_steps=args.num_steps)

    # We've completed our training, so run a test evaluation on
    # some new images we haven't used before.
    test_bottlenecks, test_ground_truth = util.get_all_cached_bottlenecks(
        sess, image_lists, 'testing',
        args.bottleneck_dir, args.image_dir, jpeg_data_tensor,
        bottleneck_tensor)
    test_bottlenecks = np.array(test_bottlenecks)
    test_ground_truth = np.array(test_ground_truth)
    print("evaluating....")
    classifier.evaluate(
        test_bottlenecks.astype(np.float32), test_ground_truth, metrics=transfer_model.METRICS)

    # write the output labels file if it doesn't already exist
    if gfile.Exists(output_labels_file):
      print("Labels list file already exists; not writing.")
    else:
      output_labels = json.dumps(list(image_lists.keys()))
      with gfile.FastGFile(output_labels_file, 'w') as f:
        f.write(output_labels)

    print("\nGetting metrics...")
    '''for name in image_lists.iterkeys():
        image_dir = image_lists[name]['dir']
        paths = [ '%s/%s/%s'%(args.image_dir, image_dir, filename) for filename in image_lists[name]['testing'] ]'''
    util_plot.plot_confusion_matrix(args, classifier, test_bottlenecks.astype(np.float32), test_ground_truth, output_labels_file)

    for name in image_lists.iterkeys():
        dir = image_lists[name]['dir']
        paths = [ '%s/%s/%s'%(args.image_dir, dir, filename) for filename in image_lists[name]['testing'] ]

    add_images(sess, paths, args.model_dir)