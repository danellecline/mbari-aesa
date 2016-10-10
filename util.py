import subprocess
import os

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
