# mbari-aesa project

Code for processing AESA images using transfer learning and the Inception v3 model.
The idea here is to take human annotations of deep-sea animals from the
Autonomous Ecological Surveying of the Abyss (AESA) project, extract images from those
annotations, and using a transfer learning method to train them for automated classification.

This is test code to help evaluate the effectiveness of the transfer learning method for
deep-sea animals classification.

Both single and multi-label classification can be tested, along with various images
distortions, and learning rates.

An example tiled image this is intended to work with:
[![ Image link ](https://github.com/danellecline/mbari-aesa/raw/master/img/M56_10441297_12987348573247_resized.jpg)]

Images are either classified by group, category, feeding type, or some combination thereof.

[![ Image link ](https://github.com/danellecline/mbari-aesa/raw/master/img/category_resized.jpg)]
[![ Image link ](https://github.com/danellecline/mbari-aesa/raw/master/img/group_resized.jpg)]

This code is based on the TensorFlow example:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py

## Prerequisites

- local copy of images and associated annotations in CSV format from the AESA project
- Python version >=2.7.9
- [ImageMagick](http://www.imagemagick.org/)
- Docker

## Running

Create virtual environment with correct dependencies

    $ pip install virtualenv
    $ virtualenv venv-mbari-aesa
    $ source venv-mbari-aesa/bin/activate
    $ pip install -r requirements.txt

Check-out code

    $ git clone https://github.com/danellecline/mbari-aesa.git

Extract the training images from the raw tiles. This will extract the images to a folder called **data** in your_project_folder

    $ cd mbari-aesa
    $ python preprocess.py --in_dir /Volumes/ScratchDrive/AESA/M56 tiles/raw/ --annotation_file /Volumes/ScratchDrive/AESA/M56_Annotations_v10.csv
    
Run single/multi label models with variations in image distortions and learning rate.

    $ batch_run.sh

Results will be default be written to the directory **data/model_output_file**. To visualize these results, open them in tensorboard.

Tensorboard can be launched from the Docker image:

    $ docker run -v /your_project_folder/mbari-aesa/data/:/tmp/ -it -p 6006:6006 -p 8888:8888 gcr.io/google-samples/tf-workshop:v4

this will launch and run the container, when in the container run the following to launch tensorboard

    root@1dc53d81b967:~# tensorboard --logdir /tmp/model_output_final/

then open a web browser to localhost:6006 to view the results

TODO: add information here on the confusion matrix/ROC curve location


## Developer Notes

