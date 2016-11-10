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
import pandas as pd
import sys
from collections import namedtuple
import util
import conf
import argparse

def process_command_line():
    from argparse import RawTextHelpFormatter

    examples = 'Examples:' + '\n\n'
    examples += sys.argv[0] + """--in_dir /Volumes/ScratchDrive/AESA/M56 tiles/raw/ --out_dir /tmp/data/images_all/cropped_images/
              --annotation_file /Volumes/ScratchDrive/AESA/M56_Annotations_v10.csv \n"""
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description='Extract cropped images from tiles and associated annotations',
                                     epilog=examples)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--by_category', action='store_true', help='Create directory of images by the column labeled Category')
    group.add_argument('--by_group', action='store_true', help='Create directory of images by the column labeled group')
    parser.add_argument('--in_dir', type=str, required=True, help="Path to folders of raw tiled images.")
    parser.add_argument('--by_category', type=str, required=True, help="Path to folders of raw tiled images.")
    parser.add_argument('--by_group', type=str, required=True, help="Path to folders of raw tiled images.")
    parser.add_argument('--out_dir', type=str, required=False, default=os.path.join(os.getcwd(),'cropped_images'), help="Path to store cropped images.")
    parser.add_argument('--annotation_file', type=str, required=True, default='/Volumes/ScratchDrive/AESA/M56_Annotations_v10.csv', help="Path to annotation file.")
    args = parser.parse_args()
    return args

def extract_annotation(raw_file, annotation, out_dir):
    '''
     crop image into square tile centered on the annotation and pad by 50 pixels
    :param raw_file:  path to file
    :param annotation: annotation tuple
    :param out_dir: directory to store cropped image
    :return:
    '''
    if "Length" in annotation.mtype :
        crop_pixels = int(float(annotation.measurement)) + 50
    else:
        crop_pixels = 500
    w = crop_pixels / 2

    if not os.path.exists(annotation.image_file):
        os.system('convert "%s" -crop %dx%d+%d+%d +repage -quality 100%% "%s"' % (
            raw_file, crop_pixels, crop_pixels, annotation.centerx - w, annotation.centery - w,annotation.image_file))
        print 'Creating  %s ...' % annotation.image_file

if __name__ == '__main__':
  args = process_command_line()

  util.ensure_dir(args.out_dir)

  aesa_annotation = namedtuple("Annotation", ["centerx", "centery", "category", "mtype", "measurement", "index", "image_file"])

  try:
    print 'Parsing ' + args.annotation_file
    df = pd.read_csv(args.annotation_file, sep=',')

    for index, row in df.iterrows():

      try:
        filename = os.path.join(args.in_dir, 'M56_10441297_%d.jpg' % int(row['FileName']))

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
        a = aesa_annotation(centerx=row['CentreX'], centery=row['CentreY'], category=row['Category'],
                            measurement=row['Measurement'], mtype=row['Type'], index=index, image_file=image_file)

        print 'Processing row %d filename %s annotation %s' % (index, filename, category)
        extract_annotation(filename, a, dir)

      except Exception as ex:
          print ex
          # TODO: store missing files to a log file

  except Exception as ex:
      print ex

  print 'Done'

