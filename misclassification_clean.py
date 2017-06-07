#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Reads in AESA annotation file and cleans according to misclassification file
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import sys
import argparse
import os
import fnmatch
import pandas as pd

def process_command_line():
    from argparse import RawTextHelpFormatter

    examples = 'Examples:' + '\n\n'
    examples += sys.argv[0] + """
              --csvdir /tmp/data/model_output_final/JC062_M535455_M56_75pad_refined_cnidaria/
              \n"""
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description='Remove misclassifications',
                                     epilog=examples)
    parser.add_argument('--csvdir', type=str, required=True, help="Path to directories with misclassified csv files.")
    parser.add_argument('--class_actual', type=str, required=True, default='', help="Classes to remove")
    parser.add_argument('--class_predicted', type=str, required=True, default='', help="Classes to remove")
    args = parser.parse_args()
    return args

#  Cleaned with:
# /Users/dcline/anaconda/bin/python /Users/dcline/Dropbox/GitHub/mbari-aesa/misclassification_clean.py --csvdir --csvdir /Users/dcline/Dropbox/GitHub/mbari-aesa/data/model_output_final/JC062_M535455_M56_75pad_refined_cnidaria/
if __name__ == '__main__':
  args = process_command_line()

  try:
    print 'Parsing ' + args.csvdir
    matches = []
    for root, dirnames, filenames in os.walk(args.csvdir):
      for filename in fnmatch.filter(filenames, 'misclassified.csv'):
        matches.append(os.path.join(root, filename))

    for m in matches:
      df = pd.read_csv(m, sep=',')
      print 'Reading %s' % m

      for index, row in df.iterrows():

        file = row['Filename']
        class_predicted = row['Predicted']
        class_actual = row['Actual']

        if class_predicted == args.class_predicted and class_actual == args.class_actual and os.path.exists(file):
          os.remove(file)
          print 'Removing {0}'.format(file)

  except Exception as ex:
      print ex


  print 'Done'

