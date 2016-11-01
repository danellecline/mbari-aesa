#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Reads in wav training file and extracts raw and averaged features over the call
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import numpy as np
import fnmatch
import os
import util
from sklearn import preprocessing
class Process:

    def __init__(self):
        print 'init'

    def extract_annotation(self, raw_file, annotation, out_dir):


        # crop image into square tile centered on the annotation and pad by 50 pixels
        if "Length" in annotation.mtype :
            crop_pixels = int(float(annotation.measurement)) + 50
        else:
            crop_pixels = 500
        w = crop_pixels / 2

        if not os.path.exists(annotation.image_file):
            os.system('convert "%s" -crop %dx%d+%d+%d +repage -quality 100%% "%s"' % (
                raw_file, crop_pixels, crop_pixels, annotation.centerx - w, annotation.centery - w,annotation.image_file))
            print 'Creating  %s ...' % annotation.image_file


    '''def extract_features(self, wav_dir):
        for root, dirnames, filenames in os.walk(wav_dir):
            for filename in fnmatch.filter(filenames, '*.jpg'):

                print 'Extracting features from %s' % filename
                match = os.path.join(root, filename)

                # pre-process with a log-polar transform



                [Fs, x] = wav.read(match)
                features = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);

                #########################
                # save raw features
                #########################
                feature_dir = os.path.join(root, 'raw_features')
                s = filename.split('.')

                # create output directory if not already created
                if not os.path.isdir(feature_dir):
                    os.mkdir(feature_dir)

                complete_path = os.path.join(feature_dir, s[0] +'.csv')
                np.savetxt(complete_path, features, delimiter=",")


                #########################
                # save average features
                #########################
                feature_dir = os.path.join(root, 'avg_features')
                s = filename.split('.')

                # create output directory if not already created
                if not os.path.isdir(feature_dir):
                    os.mkdir(feature_dir)

                avgfeatures = np.mean(features,axis = 1)
                complete_path  = os.path.join(feature_dir, 'avg' + s[0] +'.csv')
                np.savetxt(complete_path, avgfeatures, delimiter=",")'''



