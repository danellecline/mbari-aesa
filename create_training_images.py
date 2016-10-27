#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Reads in AESA annotation file and extracts training images for targeted analysis
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
import util
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':

    csv_file_raw = '/Volumes/ScratchDrive/AESA/M56_Annotations_v10.csv'
    csv_file_appended = '/Volumes/ScratchDrive/AESA/M56_Annotations_v10_appended.csv'
    image_dir = "/Volumes/ScratchDrive/AESA/M56 tiles/raw/"
    out_dir = "/Volumes/ScratchDrive/AESA/M56 tiles/raw/cropped_images/"

    aesa_annotation = namedtuple("Annotation", ["centerx", "centery", "category", "mtype", "measurement", "index", "image_file"])

    util.ensure_dir(out_dir)
    try:
        print 'Parsing ' + csv_file_raw
        df = pd.read_csv(csv_file_raw, sep=',')

        print 'Getting images from ' + image_dir
        image_files = sorted(glob.glob(image_dir +'*.jpg'))

        p = process.Process()
        for index, row in df.iterrows():

            try:
                filename = os.path.join(image_dir, 'M56_10441297_%d.jpg' % int(row['FileName']))

                # get image height and width of raw tile
                height, width = util.get_dims(filename)
                head, tail = os.path.split(filename)
                stem = tail.split('.')[0]

                # create separate directory for each category
                category = row['Category']
                dir = ('%s%s/' % (out_dir, category.upper()))
                util.ensure_dir(dir)

                image_file = '%s%06d.jpg' % (dir, index)
                a = aesa_annotation(centerx=row['CentreX'], centery=row['CentreY'], category=row['Category'],
                                    measurement=row['Measurement'], mtype=row['Type'], index=index, image_file=image_file)

                print 'Processing row %d filename %s annotation %s' % (index, filename, category)
                p.extract_annotation(filename, a, dir)

            except Exception as ex:
                print ex

    except Exception as ex:
        print ex

    print 'Done'
