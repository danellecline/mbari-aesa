#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Reads in AESA annotation file and extracts training images organized by category or group for targeted analysis
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import os
import logging
import math
import pandas as pd
import sys
from collections import namedtuple
import util
import conf
import argparse

def process_command_line():
    from argparse import RawTextHelpFormatter

    examples = 'Examples:' + '\n\n'
    examples += sys.argv[0] + """--in_dir /Volumes/ScratchDrive/AESA/M56 tiles/raw/ --by_category
              --out_dir /tmp/data/images_category/cropped_images/
              --annotation_file /Volumes/ScratchDrive/AESA/M56_Annotations_v10.csv \n"""
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description='Extract cropped images from tiles and associated annotations',
                                     epilog=examples)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--by_category', action='store_true', help='Create directory of images by the column labeled Category')
    group.add_argument('--by_group', action='store_true', help='Create directory of images by the column labeled group')
    parser.add_argument('--in_dir', type=str, required=True, help="Path to folders of raw tiled images.")
    parser.add_argument('--out_dir', type=str, required=False, default=os.path.join(os.getcwd(),'/cropped_images'), help="Path to store cropped images.")
    parser.add_argument('--annotation_file', type=str, required=True, default='/Volumes/ScratchDrive/AESA/M56_Annotations_v10.csv', help="Path to annotation file.")
    parser.add_argument('--file_format', type=str, required=False, help="Alternative file prefix to use for calculating the associated frame the annotation is from, e.g. M56_10441297_%d.jpg'")
    parser.add_argument('--strip_trailing_dot', action='store_true', required=False, help="Strip trailing .'")
    args = parser.parse_args()
    return args

def extract_annotation(raw_file, annotation):
    '''
     crop image into square tile centered on the annotation and pad by 75 pixels
    :param raw_file:  path to file
    :param annotation: annotation tuple
    :param out_dir: directory to store cropped image
    :return:
    '''
    if "Length" in annotation.mtype and not math.isnan(annotation.measurement):
        crop_pixels = int(float(annotation.measurement)) + 75
    else:
        crop_pixels = 500
    w = crop_pixels / 2

    os.system('convert "%s" -crop %dx%d+%d+%d +repage -quality 100%% "%s"' % (
        raw_file, crop_pixels, crop_pixels, annotation.centerx - w, annotation.centery - w,annotation.image_file))
    print 'Creating  %s ...' % annotation.image_file

# /Users/dcline/anaconda/bin/python /Users/dcline/Dropbox/GitHub/mbari-aesa/preprocess.py --by_category --out_dir /Users/dcline/Dropbox/GitHub/mbari-aesa/data/JC062_75pad/images_category/cropped_images/ --in_dir "/Volumes/ScratchDrive/AESA/JC062/" --annotation_file /Users/dcline/Dropbox/GitHub/mbari-aesa/data/JC062_annotations_for_Danelle.csv
# /Users/dcline/anaconda/bin/python preprocess.py --by_group --out_dir /Users/dcline/Dropbox/GitHub/mbari-aesa/data/training_images/M56_75pad/images_group/cropped_images/ --in_dir "/Volumes/ScratchDrive/AESA/M56 tiles/raw" --annotation_file /Users/dcline/Dropbox/GitHub/mbari-aesa/data/annotations/M56_Annotations_v10.csv --strip_trailing_dot --file_format M56_10441297_%d.jpg

if __name__ == '__main__':
  args = process_command_line()

  util.ensure_dir(args.out_dir)
  failed_file = open(os.path.join(args.out_dir, 'failed_crops.txt'), 'w')

  aesa_annotation = namedtuple("Annotation", ["centerx", "centery", "mtype", "measurement", "index", "image_file"])

  try:
    print 'Parsing ' + args.annotation_file
    df = pd.read_csv(args.annotation_file, sep=',')

    for index, row in df.iterrows():

      try:
        f = row['FileName']
        if args.strip_trailing_dot and isinstance(f, basestring):
          f = f.replace('.','') # handle last . sometimes found Filename column
        if args.file_format:
          filename = os.path.join(args.in_dir, args.file_format % f)
        else:
          filename = os.path.join(args.in_dir, f)

        # get image height and width of raw tile
        height, width = util.get_dims(filename)
        head, tail = os.path.split(filename)
        stem = tail.split('.')[0]

        # create separate directory for each category or group
        if args.by_category:
          category = row['Category']
          dir = ('%s%s/' % (args.out_dir, category.upper()))
          util.ensure_dir(dir)
        elif args.by_group:
          group = row['group']
          dir = ('%s%s/' % (args.out_dir, group.upper()))
          util.ensure_dir(dir)
        else:
          # default to by Category
          category = row['Category']
          dir = ('%s%s/' % (args.out_dir, category.upper()))
          util.ensure_dir(dir)

        image_file = '%s%06d.jpg' % (dir, index)
        if not os.path.exists(image_file):
          if args.by_category:
            print 'Processing row %d filename %s annotation %s' % (index, filename, category)
          elif args.by_group:
            print 'Processing row %d filename %s annotation %s' % (index, filename, group)

          a = aesa_annotation(centerx=row['CentreX'], centery=row['CentreY'],
                              measurement=row['Measurement'], mtype=row['Type'],
                              index=index, image_file=image_file)

          extract_annotation(filename, a)

      except Exception as ex:
          print ex
          failed_file.write("Error cropping annotation row {0} filename {1} \n".format(index, filename))

  except Exception as ex:
      print ex

  print 'Done'

