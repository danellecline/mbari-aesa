#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Runs transfer learning tests on AESA images
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import os
import subprocess

# image directory where cropped images are located; generated either by group or by category
image_category_dir = os.path.join(os.getcwd(),'data/images_category/cropped_images')
image_group_dir = os.path.join(os.getcwd(),'data/images_group/cropped_images')

# This is the directory the bottleneck features are generated; bottleneck features are generated by running each image through
# the inception model. Once these are generated, they are cached.
bottleneck_category_dir = os.path.join(os.getcwd(),'data/images_category/bottleneck')
bottleneck_group_dir = os.path.join(os.getcwd(),'data/images_group/bottleneck')

# annotation file location; annotation file used to generate cropped images with preprocess.py
annotation_file = os.path.join(os.getcwd(),'M56_Annotations_v10.csv')

all_options = '--num_steps 30000 --testing_percentage 30 --exclude_unknown --exclude_partial --annotation_file %s' % annotation_file
distortion_map = {
              '--learning_rate .1': 'default_p1_learning_rate',
              '--learning_rate 1' : 'default_1_learning_rate',
              '--learning_rate 10' : 'default_10_learning_rate',
              '--flip_left_right' : 'flip_left_right',
              '--rotate90' : 'rotate90',
              '--random_scale 10' : 'random_scale10',
              '--random_crop 10' : 'random_crop10',
              '--random_brightness 10' : 'random_brightness10',
              '--random_scale 20' : 'random_scale20',
              '--random_crop 20' : 'random_crop20',
              '--random_brightness 20' : 'random_brightness20',
              '--random_scale 50' : 'random_scale50',
              '--random_crop 50' : 'random_crop50',
              '--random_brightness 50' : 'random_brightness50'
              }

model_map = {
            '--exclude_unknown --image_dir {0} --bottleneck_dir {1}'.format(image_category_dir, bottleneck_category_dir):
              'category_sans_unk',
            '--exclude_partial --image_dir {0} --bottleneck_dir {1}'.format(image_category_dir, bottleneck_category_dir):
              'category_sans_partials',
            '--exclude_unknown --image_dir {0} --bottleneck_dir {1}'.format(image_group_dir, bottleneck_group_dir):
              'category_sans_unk',
            '--exclude_partial --image_dir {0} --bottleneck_dir {1}'.format(image_group_dir, bottleneck_group_dir):
              'category_sans_partials'
            }

model_map_multilabel = {
            '--multilabel_category_group':'multilabel_category_group',
            '--multilabel_group_feedingtype':'multilabel_group_feedingtype',
            '--multilabel_category_group --exclude_unknown':'multilabel_category_group_sans_unk',
            '--multilabel_category_group --exclude_partial':'multilabel_category_group_sans_partial',
            '--multilabel_category_group --exclude_partial --exclude_unknown':'multilabel_category_group_sans_partial_unk',
            '--multilabel_group_feedingtype --exclude_unknown':'multilabel_group_feedingtype_sans_unk',
            '--multilabel_group_feedingtype --exclude_partial':'multilabel_group_feedingtype_sans_partial',
            '--multilabel_group_feedingtype --exclude_partial --exclude_unknown':'multilabel_group_feedingtype_sans_partial_unk'
}

model_out_dir = os.getcwd()
for option_model,model_sub_dir in model_map.iteritems():
  for option_distort,distort_sub_dir in distortion_map.iteritems():
    cmd = 'python ./learn.py {0} {1} {2} --model_dir {3}/{4}/{5}'.format(all_options, option_model, option_distort,
                                                                         model_out_dir, model_sub_dir,distort_sub_dir)
    print(cmd)
    subproc = subprocess.Popen(cmd, env=os.environ, shell=True)
    subproc.communicate()

for option_model,model_sub_dir in model_map_multilabel.iteritems():
  for option_distort,distort_sub_dir in distortion_map.iteritems():
    cmd = 'python ./learn.py {0} {1} {2} --model_dir {3}/{4}/{5}'.format(all_options, option_model, option_distort,
                                                                         model_out_dir, model_out_dir,distort_sub_dir)
    print(cmd)
    subproc = subprocess.Popen(cmd, env=os.environ, shell=True)
    subproc.communicate()
