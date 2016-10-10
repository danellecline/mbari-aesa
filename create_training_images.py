#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Reads in VARS annotation file and extracts concept names and time codes for targeted analysis
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

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
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':

    csv_file = '/Users/dcline/Dropbox/GitHub/mbari-aesa/M56_annotations.txt'
    image_dir = '/Users/dcline/Downloads/M56/images/full/'
    out_dir = '/Users/dcline/Downloads/M56/images/output/training_images/'

    p = process();
    aesa_annotation = namedtuple("Annotation", ["filename", "centerx", "centery", "category","mtype", "measurement", "index"])

    try:
        print 'Parsing ' + csv_file
        df = pd.read_csv(csv_file, sep=',')
        d = csv_file.split('.')

        print 'Getting images from ' + image_dir
        image_files = sorted(glob.glob(image_dir +'*.jpg'))

        file_name = df['FileName']
        center_x = df['CentreX']
        center_y = df['CentreY']
        category = df['Category']
        mtype = df['Type']
        measurement = df['Measurement']

        root = 'M56'

        for name in image_files:
            print 'Searching for %s ...' % name
            indexes = [i for i in range(len(file_name)) if name in str(file_name[i]) ]

            annotations = []
            filename = os.path.join(image_dir, file_name[0])

            for i in indexes:
                f = aesa_annotation(centerx=center_x[i], centery=center_y[i], category=category[i], type=type[i],
                                    measurement=measurement[i], mtype=type[i], index=i)
                annotations.append(f)

            process.extract_annotations(filename, annotations, out_dir)

    except Exception as ex:
        print ex

    print 'Done'
