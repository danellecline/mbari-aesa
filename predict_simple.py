#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''
Loads classifiers, runs on test set and plots output
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''
from sklearn import svm, preprocessing
import numpy as np
import pandas as pd
import conf
import pickle

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from skimage.io import imread
from skimage.feature import hog
from skimage import color

if __name__ == '__main__':

    with open('classifier.pkl', 'rb') as input:
        classifier_rfc = pickle.load(input)
        print(classifier_rfc.name)

        df = pd.read_csv(conf.ANNOTATIONS_FILE, sep=',')
        ids = np.load(conf.TRAIN_IDS)
        # get the dataframes at the indexes
        df = df.ix[ids]
        predicted, expected = classifier_rfc.predict(df)

    print 'done loading classifiers'
