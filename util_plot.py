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
import fnmatch
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from matplotlib.lines import Line2D
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

def find_matches(model_out_dir, glob_filter, filename):
  matches = []
  for root, dirnames, filenames in os.walk(model_out_dir):
    for filename in fnmatch.filter(filenames, filename):
      if glob_filter in root:
        matches.append(os.path.join(root, filename))

  return matches

def plot_metrics(model_out_dir, glob_filter, distortion_map):

  try:

    # find all matches
    csv_file = os.path.join(model_out_dir, glob_filter + '_metrics_all.csv')
    matches = find_matches(model_out_dir, glob_filter, 'metrics.csv')

    # combine all the high-level data into a csv file
    header_out = False
    with open(csv_file, 'w') as fout:
      for metrics_file in sorted(matches):
        if os.path.exists(metrics_file):
          with open(metrics_file) as fin:
            header = next(fin)
            if not header_out:
                fout.write(header)
                header_out = True
            for line in fin:
                fout.write(line)

    # plot in a bar chart
    df = pd.read_csv(csv_file, sep=',')
    ax = df.plot(kind='bar', title="Metrics\n" + model_out_dir, figsize=(12,10))
    ax.set_xticklabels(df.Distortion, rotation=90)
    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(model_out_dir, os.path.join(model_out_dir, glob_filter + '_metrics_all.png')), format='png', dpi=120);

    # combine all the class-level data into a csv file
    matches = find_matches(model_out_dir, glob_filter, 'metrics_by_class.csv')
    csv_file = os.path.join(model_out_dir, glob_filter + '_metrics_all_by_class.csv')
    header_out = False
    with open(csv_file, 'w') as fout:
      for metrics_file in sorted(matches):
        if os.path.exists(metrics_file):
          with open(metrics_file) as fin:
            header = next(fin)
            if not header_out:
                fout.write(header)
                header_out = True
            for line in fin:
                fout.write(line)

    # plot each distortion in a combo bar/line plot
    df_metrics = pd.read_csv(csv_file, sep=',')
    distortions = list(set(df_metrics.Distortion))
    for v in sorted(distortions):
      df = df_metrics[df_metrics['Distortion'] == v]
      if not df.empty:
        df = df.sort(['Accuracy'], ascending=False)
        ax = df[['Accuracy','Precision']].plot(kind='bar', title="Metrics by Class" + v + "\n" + model_out_dir, figsize=(12,10))
        ax2 = ax.twinx()
        ax2.plot(ax.get_xticks(), df[['NumTrainingImages']].values, linestyle='-', marker='o', linewidth=2.0)
        ax.set_xticklabels(df_metrics.Class, rotation=90)
        ax.set(xlabel='Class')
        ax2.set(ylabel='Total Training Example Images')
        plt.tight_layout()
        fig = ax.get_figure()
        fig.savefig(os.path.join(model_out_dir, glob_filter + '_metrics_all_by_class_' + v + '.png'), format='png', dpi=120);
        plt.close('all')

    # confusion matrix plot
    matches = find_matches(model_out_dir, glob_filter, 'metrics_cm.csv')
    cmap = "YlGnBu"
    for cm_file in matches:
      df = pd.read_csv(cm_file, sep=',')
      if not df.empty:
        distortion = cm_file.split('/')[-2]
        df = pd.pivot_table(df,columns='predicted', values='num', index='actual')
        df.fillna(0, inplace=True)
        index = df.index.union(df.columns)
        df = df.reindex(index=index, columns=index, fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 10));
        ax.set_title('Confusion Matrix ' + v + '\n' + os.path(cm_file))
        sns.heatmap(df, cmap=cmap, vmax=30, annot=False, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.xticks(rotation=90)
        plt.yticks(rotation=360)
        plt.tight_layout()
        plt.savefig(os.path.join(model_out_dir, glob_filter + '_metrics_cm_by_class_' + distortion + '.png'), format='png', dpi=120);
        plt.close('all')

  except:
    print 'Error aggregating/plotting metrics'
