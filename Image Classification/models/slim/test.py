from matplotlib import pyplot as plt

import numpy as np
import os
import tensorflow as tf
import urllib2
from datasets import imagenet
from nets import vgg
from preprocessing import vgg_preprocessing

checkpoints_dir = '/home/aakash-sinha/Desktop/slim-tensorflow-git/models/slim/'

slim = tf.contrib.slim

print (tf.__version__)
