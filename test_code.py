# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:57:52 2019

@author: Stefan Borkovski
"""

import os
import cv2
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob
import time

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import custom 


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

custom_WEIGHTS_PATH = "mask_rcnn_coco_traffic_light_0100.h5"  # TODO: update this path

config = custom.CustomConfig()
custom_DIR = os.path.join(ROOT_DIR, "customImages")
test_DIR = os.path.join(ROOT_DIR, "testImages")

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

# Load validation dataset
dataset = custom.CustomDataset()
dataset.load_custom(custom_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
    
# load the last model you trained
# weights_path = model.find_last()[1]

# Load weights
print("Loading weights ", custom_WEIGHTS_PATH)
model.load_weights(custom_WEIGHTS_PATH, by_name=True)

from importlib import reload # was constantly changin the visualization, so I decided to reload it instead of notebook
reload(visualize)


###         uncomment below for testing one randomly choosen image located in directory "custom_DIR"


image_id = random.choice(dataset.image_ids)

image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                      dataset.image_reference(image_id)))
    
# Run object detection
results = model.detect([image], verbose=1)
    
# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)


###         uncomment bellow for testing multiple images located in a folder directory "test_DIR"


# class_names = ['BG','red','green','transition']
# # Load a random image from the images folder
# file_names = next(os.walk(test_DIR))[2]

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# try:
#     for i in file_names[777:]:
#         #image_name = random.choice(file_names)
        
#         #start = time.time()
        
#         #image_name = 'out00001.jpeg'
        
#         image = skimage.io.imread(os.path.join(test_DIR, i))
        
#         # Run detection
#         results = model.detect([image], verbose=1)
        
#         # Visualize results
#         r = results[0]
#     #    fig = plt.Figure(figsize=(1920, 1800))
#         visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
#         plt.savefig('testImages/' + i + '.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)
#         #end = time.time()
#         plt.close("all")
#         #print(end - start)
# except Exception:
#     pass