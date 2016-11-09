__author__ = 'dcline'

# More info here: https://github.com/tensorflow/models/blob/master/slim/README.md#Pretrained

import os
import subprocess
#import tensorflow as tf
#import sys
#head, tail = os.path.split(os.path.abspath(__file__))
#sys.path.append(os.path.join(head,'models/slim/'))
#from datasets import flowers

DATA_DIR = '/tmp/data/flowers'
path, _ = os.path.split(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(path, 'data/model_output_test/inception_v3')
PYTHON_FILE = os.path.join(path, 'models/slim/train_image_classifier.py')
PYTHON_EXE = '/Users/dcline/anaconda/bin/python'
CHECKPOINT_PATH='/tmp/models/inception_v3.ckpt'

cmd = '{0} {1} ' \
      '--train_dir={2} ' \
      '--dataset_dir={3} ' \
      '--dataset_name=flowers ' \
      '--dataset_split_name=train ' \
      '--model_name=inception_v3 ' \
      '--checkpoint_path={4} ' \
      '--checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits/Logits ' \
      '--trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits/Logits '.format(PYTHON_EXE,PYTHON_FILE,TRAIN_DIR, DATA_DIR, CHECKPOINT_PATH)

#subprocess.Popen(["virtualenv1/bin/python", "my_script.py"])

subproc = subprocess.Popen(cmd, env=os.environ, shell=True)
subproc.communicate()
print('done')


# THis doesn't work

# try bazel build
# https://www.tensorflow.org/versions/r0.9/how_tos/image_retraining/index.html
