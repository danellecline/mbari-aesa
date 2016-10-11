import os
import util


class Process:

    def __init__(self):
        print 'init'

    def extract_annotations(self, raw_file, annotations, out_dir):
        # get image height and width of raw tile - only need to do this once for each image
        height, width = util.get_dims(raw_file)
        head, tail = os.path.split(raw_file)
        stem = tail.split('.')[0]

        for a in annotations:
            j = a.index

            # crop image into square tile centered on the annotation and pad by 100 pixels
            if a.mtype is "Length":
                crop_pixels = int(float(a.measurement)) + 100
            else:
                crop_pixels = 500
            w = crop_pixels / 2

            image_dir = ('%s%s/' % (out_dir, a.category.upper()))
            util.ensure_dir(image_dir)
            out_file = '%s%s_%s_%06d' % (image_dir, stem, a.category, j)
            os.system('convert "%s" -crop %dx%d+%d+%d +repage -quality 100%% "%s"' % (
                raw_file, crop_pixels, crop_pixels, a.centerx - w, a.centery - w, out_file))
