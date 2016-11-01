#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''
Runs classifiers and stores model in pickle format for later execution
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


def hog_feature(path):
    img = imread(path)
    gray_img = color.rgb2gray(img)
    features = hog(gray_img, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), feature_vector=True)
    '''
    fd, hog_image = hog(gray_img, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), feature_vector=True) # visualise=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()'''
    return features

class Classifier(object):
    def __init__(self, name, type, df, ids):
        self.name  =  name
        self.type  = type 
        print 'Generating HOG features...'
        hog_features = [hog_feature('%s/%06d.jpg' % (conf.TRAIN_DIR, id)) for id in ids]
        print 'Creating feature vector'
        df_hog = pd.DataFrame(hog_features)
        df_measurement = df_hog.join(df['Measurement'])
        df_measurement = df_measurement.fillna(value=-1)
        X = df_measurement.as_matrix()
        np.save(conf.TRAIN_HOG, X)
     
        self.scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = preprocessing.scale(X)
        Y = df['group']
        #if type is "SVM":
        #    self.classifier  =  OneVsRestClassifier(svm.SVC(probability = True,C = 100,gamma = 0.001))
        #    self.fit  =  classif ier.fit(X_scaled, Y)
        if type is "RFC":
            self.classifier  =  OneVsRestClassifier(RandomForestClassifier(n_estimators = 120))
            self.fit  =  self.classifier.fit(X_scaled, Y)

    def predict(self, df):
        hog_features = [hog_feature('%s/%06d.jpg' % (conf.TEST_DIR, id)) for id in ids]
        print 'Creating feature vector'
        df_hog = pd.DataFrame(hog_features)
        df_measurement = df_hog.join(df['Measurement'])
        df_measurement = df_measurement.fillna(value=-1)
        X = df_measurement.as_matrix()

        X_scaled = self.scaler.scale(X)
        Y = df['group']
        #if self.type is "SVM":
        #   predicted  =  self.classifier.predict(X_scaled, Y)
        predicted = []
        expected = Y
        if self.type is "RFC":
            predicted  =  self.classifier.predict(X_scaled, Y)


        return predicted, expected


if __name__ == '__main__':
    
    with open('classifier.pkl', 'wb') as output:
         
        print 'Parsing ' + conf.ANNOTATIONS_FILE
        df = pd.read_csv(conf.ANNOTATIONS_FILE, sep=',')
    
        ids = np.load(conf.TRAIN_IDS)
     
        # get the dataframes at the indexes
        df = df.ix[ids]
    
        classifier_rfc = Classifier('RFC_HOG', 'RFC', df, ids)
        pickle.dump(classifier_rfc, output, pickle.HIGHEST_PROTOCOL)
 
        #classifier_svm = Classifier('SVM_HOG', 'SVM', df, ids)
        #pickle.dump(classifier_svm, output, pickle.HIGHEST_PROTOCOL)
 

    print 'done creating classifiers'
