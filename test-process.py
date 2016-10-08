#!/usr/bin/python
__author__ = 'dcline'

import os
import glob
import shutil
import subprocess

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

    frames = 1
    filenames = sorted(glob.glob(image_dir + '*.jpg'))
    display = ' --mbari-display-results --mbari-mark-interesting=BoundingBox '
    saliency_output=' -K --mega-combo-zoom=4 '
    opts = ' --logverb=Debug --mbari-save-event-num=all --mbari-max-event-area=50000 --mbari-cache-size=0 --mbari-min-event-frames=1 --mbari-segment-algorithm-input-image=Luminance --mbari-saliency-dist=0 --mbari-saliency-input-image=Raw --mbari-tracking-mode=None --mbari-segment-algorithm=GraphCut --vc-chans=OIC --mbari-save-boring-events=False --shape-estim-mode=ConspicuityMap --use-older-version=false --ior-type=ShapeEst --maxnorm-type=FancyOne --mbari-saliency-input-image=Raw --mbari-max-evolve-msec=15000 --boring-sm-mv=1.5 --mbari-use-foa-mask-region=False --mbari-max-event-frames=%d --mbari-save-original-frame-spec=true --mbari-save-output --levelspec=2-4,3-4,2' % frames
    #-levelspec=1-3,2-4,1'
    #--levelspec=1-3,2-4,1'
    #--levelspec=0-3,2-5,2
    cnt = 0
    index = 0
    #w = 22516/8
    #h = 2343/8

    for f in filenames:
        head, tail = os.path.split(f)
        stem = tail.split('.')[0]

        # get image height and width of raw tile
        height, width = get_dims(f)

        # some tiles are lengthwise, and some are heightwise, so split along whatever is the largest dimension
        if width > 10*height:
            horiz_tiles = 20
            vert_tiles = 1
        else:
            horiz_tiles = 1
            vert_tiles = 20

        # convert each tile into smaller overlapping image tiles
        os.system('convert %s -crop %dx%d@ +repage +adjoin %st%%06d.jpg' % (f, horiz_tiles, vert_tiles, tile_dir))

        # process each overlapping image
        for i in range(0, horiz_tiles*vert_tiles - 1):
            out_stem = '%s_t%06d' % (stem, i)

            # create a mask for each tile using a simple threshold
            mask = '%smask_f%06d.jpg' % (in_dir, 0)
            os.system('convert %st%06d.jpg -threshold 10%% %s' % (tile_dir, i, mask))

            in_file = '%st%06d.jpg' % (tile_dir, i)

            # get image height and width
            height, width = get_dims(in_file)

            if width >= 100 or height > 100:
                display_width = 640
                display_height = 480
            else:
                display_height = height
                display_width = width

            cmd = '%s %s %s %s --logverb=Info --mbari-se-size=20 --mbari-rescale-saliency=%dx%d ' \
                  '--in=%s --out=raster:%s --mbari-save-output --mbari-save-events-xml=%s.xml ' \
                  '--mbari-save-event-summary=%s.txt --mbari-rescale-display=%dx%d ' \
                  '--mbari-mark-interesting=Outline --mbari-mask-path=%s ' \
            % (bin, opts, display, saliency_output, width, height, in_file, out_stem, out_stem, out_stem,
               display_width, display_height, mask)

            if width >= 1920 or height > 1920:
                cmd = cmd + ' --mbari-segment-graph-parameters=0.95,500,250 '

            os.system(cmd)
            os.system('convert %s-results%06d.pnm %s%sf-results%06d.jpg' % (out_stem, 0, out_dir, out_stem, i))
            os.system('convert %s-T%06d.pnm %s%sf-saliency-map%06d.jpg' % ( out_stem, 0, out_dir, out_stem, i))

    exit(-1)

    # For testing
    for f in filenames:
        #os.system('convert %s %sf%06d.png' % (f, in_dir, index ))
        index += 1

    index -= 1
    cmd = '%s %s %s %s --logverb=Info  --input-frames=0-%d@1 --in=raster:%sf#.png --out=raster:%sf  --mbari-save-output \
    --mbari-save-events-xml=events.xml --mbari-save-event-summary=summary.txt --mbari-min-event-area=1000 --mbari-max-event-area=80000' % (bin, opts, display, saliency_output, index, in_dir, out_dir)
    os.system(cmd)
