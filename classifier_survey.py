#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''
Reads in HOG features and runs range of classifiers and saves resulting performance plots
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import matplotlib
matplotlib.use("Agg")
import numpy as np
import os
import conf
import classifier_utils
import pandas as pd

from sklearn import preprocessing
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit


if __name__ == '__main__':

    train_sizes = [10000, 30000]

    df = pd.read_csv(conf.ANNOTATIONS_FILE, sep=',')
    ids = np.load(conf.ALL_IDS)
    # get the dataframes at the indexes
    df = df.ix[ids]
    X = np.loadtxt(conf.ALL_HOG, delimiter=',')

    preprocessing.StandardScaler().fit(X)
    X_scaled = preprocessing.scale(X)
    Y = df['group']

    title_append = 'AESA data set - all classes'

    le = preprocessing.LabelEncoder()
    le.fit(Y)
    y = le.transform(Y)

    str = 'Class Totals:'
    for i in range(0,len(le.classes_)):
        key = le.classes_[i]
        if conf.class_dict.has_key(key):
            class_name = conf.class_dict[key]
            str += ' %s/%s: %d'  % (key, class_name, np.sum(y==i))
        else:
            str += ' %s: %d'  % (key, np.sum(y==i))

    print str

    # Split the training data into three parts: a training set, a cross-validation set, and a test set
    # Training set is 80% of the samples, and test set is 20%. Use shuffle split as this is more appropriate for
    # unbalanced class set
    n_iter = 1
    sss_train = StratifiedShuffleSplit(y, test_size=0.2, random_state=0, n_iter=n_iter)

    subtitle = 'Stratified Shuffle Split 80/20, %d Iterations, %s' % (n_iter, str)

    print 'Calculating training curves'
    plt = classifier_utils.plot_training_matrix(x, y, sss_train, subtitle, train_sizes)
    filename = '%s-training-curve-sss-iter%d.png' % (title_append, n_iter)
    full_filename = os.path.join(os.getcwd(), filename)
    plt.savefig(full_filename, dpi=120)
    print 'Saving plots to ' + full_filename

    print 'Calculating confusion matrices'
    plt = classifier_utils.plot_confusion_matrix(x, y, sss_train, subtitle, le.classes_)
    filename = '%s-confusion-matrix-sss-iter%d.png' % (title_append, n_iter)
    full_filename = os.path.join(os.getcwd(), filename)
    plt.savefig(full_filename, dpi=120)
    print 'Saving plots to ' + full_filename

    print 'done'
