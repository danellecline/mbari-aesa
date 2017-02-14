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
import pandas as pd
import sys
import util
import shutil
import argparse

def process_command_line():
    from argparse import RawTextHelpFormatter

    examples = 'Examples:' + '\n\n'
    examples += sys.argv[0] + """
              --category_dir /tmp/data/images_category/cropped_images/
              --group_dir /tmp/data/images_group/cropped_images/
              --clean_file /Volumes/ScratchDrive/AESA/M56_Annotations_v10.csv
              --annotation_file /Volumes/ScratchDrive/AESA/M56_Annotations_QAQC.csv
              \n"""
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description='Extract cropped images from tiles and associated annotations',
                                     epilog=examples)
    parser.add_argument('--category_dir', type=str, required=False, default=os.path.join(os.getcwd(),'data/M56_75pad/images_category/cropped_images'), help="Path to store cropped images.")
    parser.add_argument('--group_dir', type=str, required=False, help="Path to store cropped images.")
    parser.add_argument('--clean_file', type=str, required=False, default='M56_Annotations_QAQC.csv', help="Path to annotation file.")
    parser.add_argument('--annotation_file', type=str, required=False, default='M56_Annotations_v10.csv', help="Path to annotation file.")
    args = parser.parse_args()
    return args

#  Cleaned with:
# /Users/dcline/anaconda/bin/python /Users/dcline/Dropbox/GitHub/mbari-aesa/clean.py --category_dir /Users/dcline/Dropbox/GitHub/mbari-aesa/data/JC062_75pad/images_category/cropped_images/ --annotation_file /Users/dcline/Dropbox/GitHub/mbari-aesa/data/JC062_annotations_for_Danelle.csv --clean_file /Users/dcline/Dropbox/GitHub/mbari-aesa/data/JC062_qaqc.csv
# /Users/dcline/anaconda/bin/python /Users/dcline/Dropbox/GitHub/mbari-aesa/clean.py --category_dir /Users/dcline/Dropbox/GitHub/mbari-aesa/data/M56_75pad/images_category/cropped_images/ --annotation_file /Users/dcline/Dropbox/GitHub/mbari-aesa/data/M56_Annotations_v10.csv --clean_file /Users/dcline/Dropbox/GitHub/mbari-aesa/data/M56_Annotations_QAQC.csv
if __name__ == '__main__':
  args = process_command_line()

  try:
    print 'Parsing ' + args.clean_file
    df_clean = pd.read_csv(args.clean_file, sep=',')
    df_annotation = pd.read_csv(args.annotation_file, sep=',')

    category_group_map = {
      "CIRRIPEDIA":"CRUSTACEA",
      "ENYPNIASTESEXIMIA":"HOLOTHUROIDEA",
      "PUCNOGONIDA":"ARTHROPODA",
      "OPHIUROIDEA":"OPHIUROIDEA",
      "HOLOTHURIAN":"HOLOTHUROIDEA",
      "AMPERIMA":"HOLOTHUROIDEA",
      "HOLOTHURIAN4":"HOLOTHUROIDEA",
      "PORIFERA":"PORIFERA",
      "HOLOTHURIAN5":"HOLOTHUROIDEA",
      "PENIAGONE":"HOLOTHUROIDEA",
      "CNIDARIA2":"CNIDARIA",
      "CNIDARIA8":"CNIDARIA",
      "PENIAGONE2":"HOLOTHUROIDEA",
      "PYCNOGONIDA":"ANTHROPODA",
      "STALKED CRINOID":"CRINOIDEA",
      "TUNICATA2":"TUNICATA",
      "UMBELLULA1":"CNIDARIA",
      "CNIDARIA9":"CNIDARIA",
      "CNIDARIA10":"CNIDARIA",
      "CNIDARIA13":"CNIDARIA",
      "CNIDARIA20":"CNIDARIA",
      "CNIDARIA5":"CNIDARIA",
      "CNIDARIA7":"CNIDARIA",
      "CNIDARIA":"CNIDARIA",
      "CNIDARIA15":"CNIDARIA",
      "CNIDARIA14":"CNIDARIA",
      "CNIDARIA12":"CNIDARIA",
      "DEIMA": "HOLOTHUROIDEA",
      "UNKNOWNX":"UNKNOWN",
      "UNKNOWN":"UNKNOWN",
      "PSEUDOSTICHOPUSAEMULATUS":"HOLOTHUROIDEA",
      "PSEUDOSTICHOPUSVILLOSUS":"HOLOTHUROIDEA",
      "PSYCHROPOTESLONGICAUDA":"HOLOTHUROIDEA",
      "OPHIUROIDEAR":"OPHIUROIDEA",
      "HOLOTHURIAN6":"HOLOTHUROIDEA",
      "STALKEDCRINOID":"CRINOIDEA",
      "POLYNOID2":"POLYCHAETA",
      "FORAMINIFERA":"FORAMINIFERA",
      "MEDUSA5":"CNIDARIA"}

    has_group = False

    if args.group_dir:
      has_group = True

    for index, row in df_clean.iterrows():

      category_label = row['Morphotype'].upper()
      index_clean = int(row['CropNo'].split('.')[0])
      reassigned_category = row['Reassigned.value'].upper()
      category = df_annotation.iloc[index_clean].Category.upper()
      if has_group:
        group = df_annotation.iloc[index_clean].group.upper()

      image_file_category = '%s/%s/%06d.jpg' % (args.category_dir, category, index_clean)

      if has_group:
        group = df_annotation.iloc[index_clean].group.upper()
        image_file_group = '%s/%s/%06d.jpg' % (args.group_dir, group, index_clean)

      if reassigned_category.upper() == "REMOVE":
        if os.path.exists(image_file_category):
          os.remove(image_file_category)
        if has_group and os.path.exists(image_file_group):
          os.remove(image_file_group)
      else:
        if has_group and os.path.exists(image_file_group):
          if reassigned_category in category_group_map.keys():
            dir_group = '%s/%s' % (args.group_dir, category_group_map[reassigned_category])
            util.ensure_dir(dir_group)

            # copy file from original group to reassigned group
            dst = '%s/%s/%06d.jpg' % (args.group_dir, category_group_map[reassigned_category], index_clean)
            shutil.copyfile(image_file_category, dst)

            # remove from original group
            os.remove(image_file_group)
          else:
            print('Cannot find %s in category_group_map' % reassigned_category)
            exit(-1)

        if os.path.exists(image_file_category):
          dir_category = '%s/%s' % (args.category_dir, reassigned_category)
          util.ensure_dir(dir_category)

          # copy file from original category to reassigned category
          dst = '%s/%s/%06d.jpg' % (args.category_dir, reassigned_category, index_clean)
          shutil.copyfile(image_file_category, dst)

          # remove from original category
          os.remove(image_file_category)


  except Exception as ex:
      print ex


  print 'Done'

