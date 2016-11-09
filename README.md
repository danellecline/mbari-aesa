# mbari-aesa project

Code for processing AESA images using transfer learning and the Inception v3 model

## Prerequisites

- local copy of the images from the AESA project
- [TensorFlow](https://www.tensorflow.org/)
- Python version 2.7
- [ImageMagick](http://www.imagemagick.org/)


## Running

Create virtual environment with correct dependencies

    $ pip install virtualenv
    $ cd your_project_folder
    $ virtualenv venv-mbari-aesa
    $ TODO: python dependencies here
    
First extract the training images from the raw tiles, e.g. 

This will extract the images to a folder called "cropped_images" in your_project_folder

    $ cd your_project_folder
    $ python preprocess.py --in_dir /Volumes/ScratchDrive/AESA/M56 tiles/raw/ --annotation_file /Volumes/ScratchDrive/AESA/M56_Annotations_v10.csv