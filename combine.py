import os
import util
import fnmatch
from shutil import copyfile

prefix_dirs = ['JC062_75pad', 'M535455_75pad', 'M56_75pad']
dest_dir = os.path.join(os.getcwd(),'data', 'training_images', 'JC062_M535455_M56_75pad')
util.ensure_dir(dest_dir)

for prefix in prefix_dirs:

  # image directory where cropped images are located
  image_dir = os.path.join(os.getcwd(),'data', 'training_images', prefix,'images_category','cropped_images')

  for root, dirnames, filenames in os.walk(image_dir):
    root_bneck = root.replace('cropped_images', 'bottleneck')

    for filename in fnmatch.filter(filenames, '*.jpg'):
      # prepend the file with the prefix; copy both the image and bottleneck file if available
      class_dirname = root.split('/')[-1]

      fname_src = os.path.join(root, filename)
      d = os.path.join(dest_dir, 'images_category', 'cropped_images', class_dirname)
      util.ensure_dir(d)
      fname_dst = os.path.join(d, prefix + '_' + filename)
      copyfile(fname_src, fname_dst)


      fname_src = os.path.join(root_bneck, filename + '.txt')
      d = os.path.join(dest_dir, 'images_category', 'bottleneck', class_dirname)
      util.ensure_dir(d)
      fname_dst = os.path.join(d, prefix + '_' + filename + '.txt')
      if os.path.exists(fname_src):
        copyfile(fname_src, fname_dst)

print 'Done'


