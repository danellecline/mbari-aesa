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
#matplotlib.use('Agg')

import json
import io
import os
import math
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.lines import Line2D
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, precision_score, recall_score, average_precision_score
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

def plot_metrics(args, df, classifier, bottlenecks, test_ground_truth, labels_list_file, labels_list_file_lt20):
  if gfile.Exists(labels_list_file):
        with open(labels_list_file, 'r') as lfile:
          labels_string = lfile.read()
          labels_list = json.loads(labels_string)
          print("labels list: %s" % labels_list)
  else:
    print("Labels list %s not found" % labels_list_file)
    return None

  if gfile.Exists(labels_list_file_lt20):
        with open(labels_list_file_lt20, 'r') as lfile:
          labels_string = lfile.read()
          labels_list_lt20 = json.loads(labels_string)
          print("labels list: %s" % labels_list_lt20)
  else:
    print("Labels list %s not found" % labels_list_file_lt20)
    return None

  sess = tf.Session()
  with sess.as_default():
    results_y_test = {}
    results_y_score = {}
    matrix = np.zeros([len(labels_list),len(labels_list)],int)
    predictions = classifier.predict(x=bottlenecks, as_iterable=True)
    print("============>Predictions:")
    for key in labels_list:
      results_y_test[key] = []
      results_y_score[key] = []

    y_true = np.empty(test_ground_truth.shape[0])
    y_pred = np.empty(test_ground_truth.shape[0])

    for j, p in enumerate(predictions):
      print("---------")
      predicted = int(p['index'])
      actual = int(np.argmax(test_ground_truth[j]))
      y_true[j] = actual
      y_pred[j] = predicted
      key = labels_list[actual]
      if actual == predicted:
        results_y_test[key].append(1)
      else:
        results_y_test[key].append(0)
      results_y_score[key].append(p['class_vector'][predicted])

      print("%i is predicted as %s actual class %s %i %i" % (j, labels_list[predicted], labels_list[actual], predicted, actual))
      matrix[predicted, actual] += 1

    accuracy_all = accuracy_score(y_true, y_pred)
    precision_all = precision_score(y_true, y_pred)
    f1_all = f1_score(y_true, y_pred)
    # do with open and write because numpy doesn't export a single line output and header
    with open(os.path.join(args.model_dir,'metrics.csv'), "w") as f:
      f.write("Accuracy, Precision, F1\n")
      f.write("{0:1.5f},{1:1.5f},{2:1.5f}\n".format(accuracy_all, precision_all, f1_all))

    ind = np.arange(len(labels_list))  # the x locations for the classes
    precision = precision_score(y_true, y_pred, labels=ind, average=None)
    recall = recall_score(y_true, y_pred, labels=ind, average=None)
    f1 = f1_score(y_true, y_pred, labels=ind, average=None)
    np.savetxt(os.path.join(args.model_dir,'metrics_by_class.csv'), zip(recall, precision, f1), header="Accuracy, Precision, F1", fmt='%1.5f', delimiter=",")

    print('=============>Creating Precision/Recall ')
    fig, ax = plt.subplots(figsize=(11, 11));

    width = 0.35  # the width of the bars
    max_per_plot = 25
    label_size = 7
    plt.rcParams['xtick.labelsize'] = label_size
    plt.rcParams['ytick.labelsize'] = label_size

    if len(labels_list) >= max_per_plot:
      num_plots = int(math.ceil(float(len(labels_list))/float(max_per_plot)))
    else:
      num_plots = 1

    # sort in descending order by recall and bar plot
    temp = sorted(zip(recall, precision), key=lambda x: x[0], reverse=True)
    r, p = map(list, zip(*temp))
    gs = gridspec.GridSpec(num_plots, 1)
    s = 0; a = 0
    e = min(len(precision), max_per_plot)
    for i in range(num_plots):
      ax = fig.add_subplot(gs[a])
      num_cls = e - s

      prects = ax.bar(np.arange(num_cls), p[s:e], width=width, color='b')
      rrects = ax.bar(np.arange(num_cls) + width, r[s:e], width=width, color='r')

      ax.set_ylabel('Score')
      ax.set_xlabel('Class')
      if a == 0:
        ax.set_title('Score by class\n' + args.model_dir)

      ax.set_xticks(np.arange(num_cls) + width / 2)
      ax.set_yticks(np.arange(0, 1.0, 0.20))
      ax.set_xticklabels(labels_list[s:e])

      # Rotate date labels
      for label in ax.xaxis.get_ticklabels():
          label.set_rotation(20)
      ax.legend((prects[0], rrects[0]), ('Precision', 'Recall'))

      s = min(len(precision), e)
      e = min(len(precision), s + max_per_plot)
      a += 1

    plt.savefig(os.path.join(args.model_dir, 'metrics_prec_recall_plot.png'), format='png', dpi=120);
    plt.close()

    # compute ROC per each class and plot curve per each class
    print('============>Creating ROC and CM Plots')
    plt.close('all')
    fig = plt.figure(figsize=(11, 17));

    writer = tf.train.SummaryWriter(args.model_dir)
    gs = gridspec.GridSpec(num_plots, 1)
    a = 0
    j = 0
    for count, name in enumerate(labels_list, 1):
      color = colors[count % len(colors)]
      style = linestyles[count % len(linestyles)]
      marker = markers[count % len(markers)]

      # if one of the classes with less than 20 images, enlarge the marker a little
      if name in labels_list_lt20:
        marker_size = 10
      else:
        marker_size = 5

      if (count + max_per_plot) % max_per_plot  == 1:
        print('Adding subplot %d numplots %d number of labels %d' % (a, num_plots, len(labels_list)))
        ax = fig.add_subplot(gs[a])
        ax.set_title('ROC Curves \n' + args.model_dir)
        ax.set_xlim([0, 1.0])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        a += 1

      if not results_y_test[name] and not results_y_score[name]:
        print('===========> Warning - no values assigned to %s <=======' % name)
        ax.annotate('unassigned category ' + name, xy=(1, 0 + j), xycoords='axes fraction', horizontalalignment='right',
            verticalalignment='bottom', fontsize=5)
        j += .02
      else:
        print('===========> Adding to plot %s <=======' % name)
        # compute ROC curve, area, and F1 score for this class and plot
        fpr, tpr, _ = roc_curve(results_y_test[name], results_y_score[name])
        if np.isnan(fpr).any() or np.isnan(tpr).any():
          ax.annotate('unassigned category ' + name, xy=(1, 0 + j), xycoords='axes fraction', horizontalalignment='right',
          verticalalignment='bottom', fontsize=5)
          j += .02
        else:
          roc_auc = auc(fpr, tpr)
          ax.plot(fpr, tpr, linestyle=style,  marker=marker, color=color, markersize=marker_size, label='category {0} (area = {1:0.2f})'.format(name, roc_auc));

      ax.legend(loc='best')

    plt.savefig(os.path.join(args.model_dir, 'metrics_roc_plot.png'), format='png', dpi=120);
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', dpi=120);
    buf2.seek(0)
    #plt.show()
    # Convert PNG buffer to TF image, add batch dimension and image summary; this will post the plot as an image to tensorboard
    image2 = tf.image.decode_png(buf2.getvalue(), channels=4)
    image2 = tf.expand_dims(image2, 0)
    head, tail = os.path.split(args.model_dir)
    summary_op = tf.image_summary('plt_roc', image2)
    summary = sess.run(summary_op)
    writer.add_summary(summary)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 11));
    ax.set_title('Confusion Matrix \n' + args.model_dir)

    cmap = "YlGnBu" #sns.diverging_palette(220, 10, as_cmap=True)
    # compute confusion matrix and color with heatmap
    # annotate with numbers if less than 10 labels, otherwise too cluttered
    annot = (len(labels_list) < 10)
    sns.heatmap(matrix, cmap=cmap, vmax=30, annot=annot,
          square=True, xticklabels=labels_list, yticklabels=labels_list,
          linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.savefig(os.path.join(args.model_dir, 'metrics_cm_plot.png'), format='png', dpi=120);
    plt.close()

    # save CM as a csv file
    np.savetxt(os.path.join(args.model_dir,'metrics_cm.csv'), matrix, fmt='%1.5f', delimiter=",")

    # Convert PNG buffer to TF image, add batch dimension and image summary; this will post the plot as an  image to tensorboard
    '''buf3 = io.BytesIO()
    plt.savefig(buf3, format='png', dpi=120);
    buf3.seek(0)
    image3 = tf.image.decode_png(buf3.getvalue(), channels=4)
    image3 = tf.expand_dims(image3, 0)
    head, tail = os.path.split(args.model_dir)
    summary_op3 = tf.image_summary('plt_confusion_matrix', image3)
    summary3 = sess.run(summary_op3)
    writer.add_summary(summary3)
    writer.close()'''
    print('Done')
