#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Utility class for miscellaneous learning plot functions

@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

# uncomment this if you want to display plots, e.g. use the plt.show() function
import matplotlib
matplotlib.use('Agg')

import json
import io
import os

import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve
from tensorflow.python.platform import gfile

linestyles = ['-', '--', ':']
markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

styles = markers + [
    r'$\lambda$',
    r'$\bowtie$',
    r'$\circlearrowleft$',
    r'$\clubsuit$',
    r'$\checkmark$']

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

def plot_confusion_matrix(args, classifier, bottlenecks, test_ground_truth, labels_list_file):
  if gfile.Exists(labels_list_file):
        with open(labels_list_file, 'r') as lfile:
          labels_string = lfile.read()
          labels_list = json.loads(labels_string)
          print("labels list: %s" % labels_list)
  else:
    print("Labels list %s not found" % labels_list_file)
    return None

  sess = tf.Session()
  with sess.as_default():
    results_y_test = {}
    results_y_score = {}
    precision = {}
    recall =  {}
    average_precision =  {}
    matrix = np.zeros([len(labels_list),len(labels_list)],int)
    predictions = classifier.predict(x=bottlenecks, as_iterable=True)
    print("============>Predictions:")
    j = 0
    for key in labels_list:
      results_y_test[key] = []
      results_y_score[key] = []
      precision[key] = []
      recall[key] = []
      average_precision[key] = []

    for _, p in enumerate(predictions):
      print("---------")
      predicted = int(p['index'])
      actual = int(np.argmax(test_ground_truth[j]))
      key = labels_list[actual]
      if actual == predicted:
        results_y_test[key].append(0)
      else:
        results_y_test[key].append(1)
      results_y_score[key].append(p['class_vector'][actual])

      #print("%s is predicted as %s actual class %s vector %s " % (test_paths[j], labels_list[predicted], labels_list[actual], p['class_vector']))
      print("%i is predicted as %s actual class %s vector %s " % (j, labels_list[predicted], labels_list[actual], p['class_vector']))
      matrix[predicted, actual] += 1
      j += 1

    max_per_plot = 15
    if len(labels_list) > max_per_plot:
      num_plots = int(len(labels_list)/max_per_plot)
    else:
      num_plots = 1

    with sns.color_palette("husl", 100):
      fig = plt.figure(figsize=(11, 17));
      sns.set()
      writer = tf.train.SummaryWriter(args.model_dir)
      gs = gridspec.GridSpec(num_plots, 1)
      # compute Precision-Recall per each class and plot curve per each class
      print('============>Creating ROC and CM Plots')
      a = 0
      for count, name in enumerate(labels_list, 1):
          if not results_y_test[name] and not results_y_score[name]:
            print('===========> Warning - no values assigned to %s <=======' % name)
            results_y_test[name].append(0)
            results_y_score[name].append(0)

          precision[name], recall[name], _ = precision_recall_curve(results_y_test[name], results_y_score[name])

          color = colors[count % len(colors)]
          style = linestyles[count % len(linestyles)]
          marker = markers[count % len(markers)]
          if (count + max_per_plot) % max_per_plot  == 1:
            ax = fig.add_subplot(gs[a])
            ax.set_title('ROC Curves')
            a += 1
          if count % max_per_plot  == 0:
            ax.legend(loc="lower left")

          ax.plot(recall[name], precision[name], linestyle=style,  marker=marker, color=color, markersize=5,
             label='Precision-recall curve of class {0}'
                   ''.format(name));

      '''avg_f_score = np.average(results.f_score)
      avg_precision = np.average(results.precision)
      avg_accuracy = np.average(results.accuracy)
      annotation = '(averages) f_score:{0:0.2f} precision:{1:0.2f} accuracy:{2:0.2f}'\
                      .format(avg_f_score, avg_precision, avg_accuracy)
      ax2.annotate(annotation, xy=(1, 0), xycoords='axes fraction', horizontalalignment='right',
                  verticalalignment='bottom', fontsize=10)
      ax.legend(loc="lower left") '''

      plt.savefig(os.path.join(args.model_dir, 'm_roc_' + args.metrics_plot_name), format='png', dpi=120);
      buf2 = io.BytesIO()
      plt.savefig(buf2, format='png', dpi=120);
      buf2.seek(0)
      #plt.show()
      # Convert PNG buffer to TF image, add batch dimension and image summary; this will post the plot as an  image to tensorboard
      image2 = tf.image.decode_png(buf2.getvalue(), channels=4)
      image2 = tf.expand_dims(image2, 0)
      head, tail = os.path.split(args.model_dir)
      tag = '{0}/m_roc_{1}'.format(tail,args.metrics_plot_name)
      summary_op = tf.image_summary(tag, image2)
      summary = sess.run(summary_op)
      writer.add_summary(summary)

      plt.close('all')

      fig, ax = plt.subplots(figsize=(5, 5));
      ax.set_title('Confusion Matrix ')
      # compute confusion matrix and color with heatmap
      sns.heatmap(matrix, ax=ax, annot=True, fmt='d', linewidths=.5, yticklabels=labels_list, xticklabels=labels_list)
      plt.savefig(os.path.join(args.model_dir, 'm_cm_' + args.metrics_plot_name), format='png', dpi=120);
      # Convert PNG buffer to TF image, add batch dimension and image summary; this will post the plot as an  image to tensorboard
      buf3 = io.BytesIO()
      plt.savefig(buf3, format='png', dpi=120);
      buf3.seek(0)
      image3 = tf.image.decode_png(buf3.getvalue(), channels=4)
      image3 = tf.expand_dims(image3, 0)
      head, tail = os.path.split(args.model_dir)
      tag = '{0}/m_cm_{1}'.format(tail, args.metrics_plot_name)
      summary_op3 = tf.image_summary(tag, image3)
      summary3 = sess.run(summary_op3)
      writer.add_summary(summary3)
      writer.close()