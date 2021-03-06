#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Iteratively runs transfer learning tests on AESA images and remove  misclassifications for certain classes.
Only those classes with general confusion across many classes were removed between each iteration.
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import os
import subprocess
import util
import util_plot
import glob

def batch_process(prefix, annotation_file, model_out_dir, options):
  distortion_map = {
    '--random_scale 20': 'random_scale_20',
    '--random_scale 10': 'random_scale_10',
    '--random_crop 10': 'random_crop_10',
    '--random_brightness 10': 'random_brightness_10',
    '--random_crop 20': 'random_crop_20',
    '--random_brightness 20': 'random_brightness_20',
    '--random_scale 50': 'random_scale_50',
    '--random_crop 50': 'random_crop_50',
    '--random_brightness 50': 'random_brightness_50'
  }


  # image directory where exemplar images are
  exemplar_dir = os.path.join(os.getcwd(),'data', 'training_images', 'exemplars_sans_JC062')

  # image directory where cropped images are located; generated either by group or by category
  image_category_dir = os.path.join(os.getcwd(),'data', 'training_images', 'optimal',  prefix,'images_category','cropped_images')

  # This is the directory the bottleneck features are generated; bottleneck features are generated by running each image through
  # the inception model. Once these are generated, they are cached.
  bottleneck_category_dir = os.path.join(os.getcwd(),'data', 'training_images', 'optimal', prefix,'images_category','bottleneck')

  model_map = { '--image_dir {0} --bottleneck_dir {1}'.format(image_category_dir, bottleneck_category_dir): 'category_sans_unk' }

  for option_model,model_sub_dir in model_map.iteritems():
    for option_distort,distort_sub_dir in distortion_map.iteritems():
      out_dir = '{0}/{1}/{2}'.format(model_out_dir, model_sub_dir, distort_sub_dir)
      util.ensure_dir(out_dir)
      all_options = ' --exemplar_dir {0} --annotation_file {1} {2}'.format(exemplar_dir, annotation_file, options)
      cmd = 'python ./learn.py {0} {1} {2} --model_dir {3}'.format(all_options, option_model, option_distort, out_dir)
      print(cmd)
      subproc = subprocess.Popen(cmd, env=os.environ, shell=True)
      subproc.communicate()

  util_plot.plot_metrics(model_out_dir, 'category_sans_unk')

if __name__ == '__main__':

  annotation_file = os.path.join(os.getcwd(),'data','annotations','M56_Annotations_v10.csv')
  options = '--num_steps 30000 --testing_percentage 30 --learning_rate .01 '
  prefix = 'M535455_75pad_cnidaria'

  model_out_dir = os.path.join(os.getcwd(),'data/model_output_final/optimal_all_clean_cnidaria2_8', prefix)
  batch_process(prefix=prefix, model_out_dir=model_out_dir, annotation_file=annotation_file, options=options)

  cmd = 'python ./misclassification_clean.py --class_actual cnidaria9 --class_predicted cnidaria2 --csvdir %s' % model_out_dir
  subproc = subprocess.Popen(cmd, env=os.environ, shell=True)
  subproc.communicate()
  cmd = 'python ./misclassification_clean.py --class_actual cnidaria8 --class_predicted cnidaria2 --csvdir %s' % model_out_dir
  subproc = subprocess.Popen(cmd, env=os.environ, shell=True)
  subproc.communicate()
  cmd = 'python ./misclassification_clean.py --class_actual cnidaria7 --class_predicted cnidaria2 --csvdir %s' % model_out_dir
  subproc = subprocess.Popen(cmd, env=os.environ, shell=True)
  subproc.communicate()
  cmd = 'python ./misclassification_clean.py --class_actual cnidaria8 --class_predicted cnidaria10 --csvdir %s' % model_out_dir
  subproc = subprocess.Popen(cmd, env=os.environ, shell=True)
  subproc.communicate()

  model_out_dir = os.path.join(os.getcwd(), 'data/model_output_final/optimal_all_clean_cnidaria2_8', prefix + '_iter1')
  batch_process(prefix=prefix, model_out_dir=model_out_dir, annotation_file=annotation_file, options=options)

  #cmd = 'python ./misclassification_clean.py --classes cnidaria2,cnidaria8 --csvdir %s' % model_out_dir
  #print(cmd)
  #subproc = subprocess.Popen(cmd, env=os.environ, shell=True)
  #subproc.communicate()

  #model_out_dir = os.path.join(os.getcwd(), 'data/model_output_final/optimal_all_clean_cnidaria2_8', prefix + '_iter2')
  #batch_process(prefix=prefix, model_out_dir=model_out_dir, annotation_file=annotation_file, options=options)

print 'Done'
