#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Reads in training data and extracts HOG features
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
import numpy as np
import glob
import os
import conf
import fnmatch
import os


def hog_feature(path):
    img = imread(path)
    gray_img = color.rgb2gray(img)
    features = hog(gray_img, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), feature_vector=True)
    return features

def extract(df, npyid, image_dir, out):

    print 'Generating HOG features...'
    ids = np.load(npyid)
    # get the dataframes at the indexes
    df = df.ix[ids]
    hog_features = [hog_feature('%s/%06d.jpg' % (image_dir, id)) for id in ids]
    print 'Creating feature vector'
    df_hog = pd.DataFrame(hog_features)
    df_measurement = df_hog.join(df['Measurement'])
    df_measurement = df_measurement.fillna(value=-1)
    X = np.array(df_measurement) #df_measurement.as_matrix()
    np.savetxt(out, X, delimiter=",")


if __name__ == '__main__':

    print 'Parsing ' + conf.ANNOTATIONS_FILE
    df = pd.read_csv(conf.ANNOTATIONS_FILE, sep=',')
    #extract(df, conf.TEST_IDS, conf.TEST_DIR, conf.TEST_HOG)
    #extract(df, conf.TRAIN_IDS, conf.TRAIN_DIR, conf.TRAIN_HOG)
    # files are named according to the index in the dataframe
    matches = []
    for root, dirnames, filenames in os.walk(conf.CROPPED_DIR):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            matches.append(os.path.join(root, filename))

    ids = [int(os.path.basename(s).replace(".jpg", "")) for s in matches]
    ids.sort()
    hog_features = [hog_feature('%s' % (match)) for match in matches]
    # get the dataframes at the indexes
    df = df.ix[ids]
    print 'Creating feature vector'
    df_hog = pd.DataFrame(hog_features)
    df_measurement = df_hog.join(df['Measurement'])
    df_measurement = df_measurement.fillna(value=-1)
    X = np.array(df_measurement) #df_measurement.as_matrix()
    np.savetxt(conf.ALL_HOG, X, delimiter=",")

    print "Saving %s" % conf.ALL_IDS
    np.save(conf.ALL_IDS, np.array(ids))

