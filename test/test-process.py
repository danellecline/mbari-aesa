#!/usr/bin/python
__author__ = 'dcline'

import os
import glob
import shutil
import subprocess
import tarfile
from shutil import copyfile
from collections import namedtuple
import math
import os
import sys
import glob
from shutil import copyfile

def get_dims(image):
    # get the height and width of a tile
    cmd = 'identify %s' % (image)
    subproc = subprocess.Popen(cmd, env=os.environ, shell=True, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    out, err = subproc.communicate()
    f = out.rstrip()
    a = f.split(' ')[3]
    size = a.split('+')[0]
    width = int(size.split('x')[0])  # /4
    height = int(size.split('x')[1])  # /4
    return height, width

def ensure_dir(fname):
    d = os.path.dirname(fname)
    if not os.path.exists(d):
        print "Making directory %s" % d
        os.makedirs(d)
if __name__ == '__main__':

    bin = '/home/avedac/dev/avedac-bitbucket/aved-mbarivision/src/main/cpp/target/build/bin/mbarivision'
    image_dir = '/mnt/hgfs/Downloads/M56/images/full/'
    out_dir = '/mnt/hgfs/Downloads/M56/output/full_graphcut_foa/'
    tile_dir = '/mnt/hgfs/Downloads/M56/input/tile/'
    in_dir = '/mnt/hgfs/Downloads/M56/input/full/'
    mask_file = '/mnt/hgfs/Downloads/M56/mask.jpg'

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)

    if not os.path.isdir(in_dir):
        os.makedirs(in_dir)

    if not os.path.isdir(tile_dir):
        os.makedirs(tile_dir)

    filenames = sorted(glob.glob(image_dir + '*.jpg'))
    annotations = []
    aesa_annotation = namedtuple("Annotation", ["filename", "centerx", "centery", "category","type", "measurement"])
    f = aesa_annotation(filename=os.path.join(image_dir, 'M56_10441297_12987348579669.jpg'), centerx=2918, centery=526, category='Cnidaria2', type='Length', measurement='40.70626')
    annotations.append(f)
    f = aesa_annotation(filename=os.path.join(image_dir, 'M56_10441297_12987348579669.jpg'), centerx=1759, centery=4337, category='Cnidaria2', type='Length', measurement='47.010646')
    annotations.append(f)
    f = aesa_annotation(filename=os.path.join(image_dir, 'M56_10441297_12987348579669.jpg'), centerx=1175, centery=10962, category='Cnidaria2', type='Length', measurement='53.36666')
    annotations.append(f)
    f = aesa_annotation(filename=os.path.join(image_dir, 'M56_10441297_12987348579669.jpg'), centerx=572, centery=11104, category='Cnidaria2', type='Length', measurement='31.400646')
    annotations.append(f)
    f = aesa_annotation(filename=os.path.join(image_dir, 'M56_10441297_12987348579669.jpg'), centerx=1689, centery=11992, category='Cnidaria2', type='Length', measurement='47.26521')
    annotations.append(f)
    f = aesa_annotation(filename=os.path.join(image_dir, 'M56_10441297_12987348579669.jpg'), centerx=1450, centery=13109, category='Cnidaria2', type='Length', measurement='19.64688')
    annotations.append(f)
    j = 0
    for a in annotations:
        head, tail = os.path.split(a.filename)
        stem = tail.split('.')[0]

        # get image height and width of raw tile
        height, width = get_dims(a.filename)

        # crop image into square tile centered on the annotation and pad by 100 pixels
        if a.type is "Length":
            crop_pixels = int(float(a.measurement)) + 100
        else:
            crop_pixels = 500
        w = crop_pixels/2

        image_dir = ('%s%s/'% (out_dir, a.category))
        ensure_dir(image_dir)
        out_file = '%s%s_%s_%06d' % (image_dir, stem, a.category, j)
        os.system('convert %s -crop %dx%d+%d+%d +repage -quality 100%% %s' % (a.filename, crop_pixels, crop_pixels, a.centerx - w, a.centery - w, out_file))

        j += 1
    exit(-1)