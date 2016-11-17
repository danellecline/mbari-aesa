#!/bin/bash
# run with nohup ./batch_run.sh > /tmp/nohup.out 2>&1
source /home/dcline-admin/Dropbox/GitHub/venv-mbari-aesa-dev-box/bin/activate
annotation_file=`pwd`/M56_Annotations_v10.csv
#python=~/anaconda/bin/python
python batch_run.py