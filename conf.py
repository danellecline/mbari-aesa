import os
#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Globals used in AESA project
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

global ANNOTATIONS_FILE, TILE_DIR, OUT_DIR

ANNOTATIONS_FILE = 'M56_Annotations_v10.csv'
#ANNOTATIONS_FILE = '/Users/dcline/Dropbox/GitHub/mbari-aesa/M56_Annotations_v10.csv'
TILE_DIR = "/Volumes/ScratchDrive/AESA/M56 tiles/raw/"
CROPPED_DIR = "/Volumes/ScratchDrive/AESA/M56 tiles/raw/cropped_images/"
TEST_IDS = "data/test_ids.npy"
ALL_IDS = "data/all_ids.npy"
TRAIN_IDS = "data/train_ids.npy"
TEST_DIR = "data/images_test/"
TRAIN_DIR = "data/images_train/"
TEST_HOG = "data/test_hog.csv"
TRAIN_HOG = "data/train_hog.csv"
ALL_HOG = "data/all_hog.csv"
