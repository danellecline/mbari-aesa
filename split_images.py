#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Reads in AESA training images and splits into training/test/validation set
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''
import shutil
import fnmatch
import os
import csv
import logging
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import subprocess
import process

from collections import namedtuple
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit, train_test_split
from shutil import copyfile
import util
import shutil

import numpy as np
import glob
import os
import conf
import fnmatch
import os

if __name__ == '__main__':

    try:
        print 'Parsing ' + conf.ANNOTATIONS_FILE
        df = pd.read_csv(conf.ANNOTATIONS_FILE, sep=',')

        # files are named according to the index in the dataframe
        matches = []
        for root, dirnames, filenames in os.walk(conf.CROPPED_DIR):
            for filename in fnmatch.filter(filenames, '*.jpg'):
                matches.append(os.path.join(root, filename))
        ids = [int(os.path.basename(s).replace(".jpg", "")) for s in matches]

        # get the dataframes at the indexes
        df_subset = df.ix[ids]

        y = df_subset.pop('group')
        X = df_subset
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y) # split 70% training/30% testing
        #x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5) # 15% for test and validation

        if os.path.exists(conf.TRAIN_DIR):
            shutil.rmtree(conf.TRAIN_DIR)
        util.ensure_dir(conf.TRAIN_DIR)

        if os.path.exists(conf.TEST_DIR):
            shutil.rmtree(conf.TEST_DIR)
        util.ensure_dir(conf.TEST_DIR)

        print 'Copying training images to ' + conf.TRAIN_DIR + ' ...'
        for index, row in x_train.iterrows():
            category = row.Category
            dir = ('%s%s/' % (conf.CROPPED_DIR, category.upper()))
            src = '%s/%06d.jpg' % (dir, index)
            dst = '%s/%06d.jpg' % (conf.TRAIN_DIR, index)
            if os.path.exists(src):
                copyfile(src, dst)

        filenames = glob.glob(conf.TRAIN_DIR + '/*.jpg')
        ids = [int(os.path.basename(s).replace(".jpg", "")) for s in filenames]
        ids.sort()
        ids = np.array(ids)
        print "Saving %s" % conf.TRAIN_IDS
        np.save(conf.TRAIN_IDS, ids)


        print 'Copying test images to ' + conf.TEST_DIR + ' ...'
        for index, row in x_test.iterrows():
            category = row.Category
            dir = ('%s%s/' % (conf.CROPPED_DIR, category.upper()))
            src = '%s/%06d.jpg' % (dir, index)
            dst = '%s/%06d.jpg' % (conf.TEST_DIR, index)
            if os.path.exists(src):
                copyfile(src, dst)

        filenames = glob.glob(conf.TEST_DIR + '/*.jpg')
        ids = [int(os.path.basename(s).replace(".jpg", "")) for s in filenames]
        ids.sort()
        ids = np.array(ids)
        print "Saving %s" % conf.TEST_IDS
        np.save(conf.TEST_IDS, ids)


    except Exception as ex:
        print ex

    print 'Done'
