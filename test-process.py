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
    display = ' --mbari-display-results --mbari-mark-interesting=BoundingBox '
    saliency_output=' -K --mega-combo-zoom=4 '
    opts = ' --logverb=Debug --mbari-save-event-num=all --mbari-max-event-area=20000 --mbari-min-event-area=100 --mbari-cache-size=0 ' \
           '--mbari-min-event-frames=1 --mbari-segment-algorithm-input-image=Luminance --mbari-saliency-dist=0 ' \
           '--mbari-saliency-input-image=Raw --mbari-tracking-mode=None --mbari-segment-algorithm=GraphCut ' \
           '--vc-chans=O:5IC  --mbari-save-boring-events=False --shape-estim-mode=ConspicuityMap ' \
           '--use-older-version=false --ior-type=ShapeEst --maxnorm-type=FancyOne --mbari-saliency-input-image=Raw ' \
           '--mbari-max-evolve-msec=2000 --boring-sm-mv=0.5 --mbari-use-foa-mask-region=False ' \
           '--mbari-max-event-frames=1 --mbari-save-original-frame-spec=true --mbari-save-output ' \
           ' --mbari-max-WTA-points=20 '
    #-levelspec=1-3,2-4,1'
    #--levelspec=1-3,2-4,1'
    #--levelspec=0-3,2-5,2
    cnt = 0
    index = 0
    #w = 22516/8
    #h = 2343/8

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

        # crop image into square tile centered on the annotation
        if a.type is "Length":
            crop_pixels = int(float(a.measurement))
        else:
            crop_pixels = 500
        w = crop_pixels/2
        os.system('convert %s -crop %dx%d+%d+%d +repage %st%06d.jpg' % (a.filename, crop_pixels, crop_pixels, a.centerx - w, a.centery - w, tile_dir, j))

        out_stem = '%s_t%06d' % (stem, j)

        # create a mask for each croppped annotation using a simple threshold
        mask = '%smask_f%06d.jpg' % (in_dir, 0)
        os.system('convert %st%06d.jpg -threshold 10%% %s' % (tile_dir, j, mask))

        in_file = '%st%06d.jpg' % (tile_dir, j)

        # get image height and width
        height, width = get_dims(in_file)

        cmd = '%s %s %s %s --logverb=Info --mbari-rescale-saliency=%dx%d ' \
              '--in=%s --out=raster:%s --mbari-save-output --mbari-save-events-xml=%s.xml ' \
              '--mbari-save-event-summary=%s.txt --mbari-rescale-display=%dx%d ' \
              '--mbari-mark-interesting=Outline --mbari-mask-path=%s ' \
              '--foa-radius=10 --fovea-radius=10 ' \
        % (bin, opts, display, saliency_output, width, height, in_file, out_stem, out_stem, out_stem,
           width, height, mask)

        os.system(cmd)
        os.system('convert %s-results%06d.pnm %s-results%06d.jpg' % (out_stem, 0, out_stem, j))
        os.remove('%s-results%06d.pnm' % (out_stem, 0))
        os.system('convert %s-T%06d.pnm %sf-saliency-map%06d.jpg' % ( out_stem, 0,out_stem, j))
        os.remove('%s-T%06d.pnm' % (out_stem, 0))
        j += 1

        tar_file = '%s.tar.gz' % out_stem
        print 'Compressing processed results in %s to %s' % (os.getcwd(), tar_file)
        tf = tarfile.open(tar_file, 'w:gz')
        for name in sorted(glob.glob(os.path.join(os.getcwd(), out_stem + '*evt*.pnm'))):
            head, tail = os.path.split(name)
            stem = tail.split('.')[0]
            out_file = os.path.join(os.getcwd(), '%s.jpg' % stem)
            if os.system('convert %s %s' % (name, out_file)) == 0:
                os.remove(name)
                print 'adding ' + out_file
                tf.add(out_file)

        tf.close()

        for name in sorted(glob.glob(os.path.join(os.getcwd(), out_stem + '*'))):
            n = os.path.basename(name)
            out = os.path.join(out_dir, n)
            print "Copying %s to directory %s" % (name, out)
            copyfile(name, out)
            os.remove(name)
        print 'foobar'
    exit(-1)