# Introduction 

In this post I want to show an example of application of Tensorflow and a recently released library slim for Image Classification, Image Annotation and Segmentation. In the post I focus on slim, cover a small theoretical part and show possible applications.

I have tried other libraries before like Caffe, Matconvnet, Theano and Torch. All of them have their pros and cons, but I always wanted a library in Python that is flexible, has good support and has a lot of pretrained models. Recently, a new library called slim was released along with a set of standart pretrained models like ResNet, VGG, Inception-ResNet-v2 (new winner of ILSVRC) and others. This library along with models are supported by Google, which makes it even better. There was a need for a library like this because Tensorflow itself is a very low-level and any implementation can become highly complicated. It requires writing a lot of boilerplate code. Reading other people’s code was also complicated. slim is a very clean and lightweight wrapper around Tensorflow with pretrained models.

This post assumes a prior knowledge of Tensorflow and Convolutional Neural Networks. Tensorflow has a nice tutorials on both of these. You can find them here.

The blog post is created using jupyter notebook. After each chunk of a code you can see the result of its evaluation. You can also get the notebook file from here. The content of the blog post is partially borrowed from slim walkthough notebook.

Setup
To be able to run the code, you will need to have Tensorflow installed. I have used r0.11. You will need to have tensorflow/models repository cloned. To clone it, simply run:

git clone https://github.com/tensorflow/models
I am also using scikit-image library and numpy for this tutorial plus other dependencies. One of the ways to install them is to download Anaconda software package for python.

First, we specify tensorflow to use the first GPU only. Be careful, by default it will use all available memory. Second, we need to add the cloned repository to the path, so that python is able to see it.

import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append("/home/dpakhom1/workspace/models/slim")
Now, let’s download the VGG-16 model which we will use for classification of images and segmentation. You can also use networks that will consume less memory(for example, AlexNet). For more models look here.

from datasets import dataset_utils
import tensorflow as tf

url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"

## Specify where you want to download the model to
checkpoints_dir = '/home/dpakhom1/checkpoints'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)
>> Downloading vgg_16_2016_08_28.tar.gz 100.0%
Successfully downloaded vgg_16_2016_08_28.tar.gz 513324920 bytes.
Image Classification
The model that we have just downloaded was trained to be able to classify images into 1000 classes. The set of classes is very diverse. In our blog post we will use the pretrained model to classify, annotate and segment images into these 1000 classes.

Below you can see an example of Image Classification. We preprocess the input image by resizing it while preserving the aspect ratio and crop the central part. The size of the crop is equal to the size of images that the network was trained on.

%matplotlib inline

from matplotlib import pyplot as plt

import numpy as np
import os
import tensorflow as tf
import urllib2

from datasets import imagenet
from nets import vgg
from preprocessing import vgg_preprocessing

checkpoints_dir = '/home/dpakhom1/checkpoints'

slim = tf.contrib.slim

### We need default size of image for a particular network.
### The network was trained on images of that size -- so we
### resize input image later in the code.
image_size = vgg.vgg_16.default_image_size


with tf.Graph().as_default():
    
    url = ("https://upload.wikimedia.org/wikipedia/commons/d/d9/"
           "First_Student_IC_school_bus_202076.jpg")
    
    ## Open specified url and load image as a string
    image_string = urllib2.urlopen(url).read()
    
    ## Decode string into matrix with intensity values
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    ## Resize the input image, preserving the aspect ratio
    ## and make a central crop of the resulted image.
    ## The crop will be of the size of the default image size of
    ## the network.
    processed_image = vgg_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)
    
    ## Networks accept images in batches.
    ## The first dimension usually represents the batch size.
    ## In our case the batch size is one.
    processed_images  = tf.expand_dims(processed_image, 0)
    
    ## Create the model, use the default arg scope to configure
    ## the batch norm parameters. arg_scope is a very conveniet
    ## feature of slim library -- you can define default
    ## parameters for layers -- like stride, padding etc.
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _ = vgg.vgg_16(processed_images,
                               num_classes=1000,
                               is_training=False)
    
    ## In order to get probabilities we apply softmax on the output.
    probabilities = tf.nn.softmax(logits)
    
    ## Create a function that reads the network weights
    ## from the checkpoint file that you downloaded.
    ## We will run it in session later.
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))
    
    with tf.Session() as sess:
        
        ## Load weights
        init_fn(sess)
        
        ## We want to get predictions, image as numpy matrix
        ## and resized and cropped piece that is actually
        ## being fed to the network.
        np_image, network_input, probabilities = sess.run([image,
                                                           processed_image,
                                                           probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                            key=lambda x:x[1])]
    
    ## Show the downloaded image
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.suptitle("Downloaded image", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()
    
    ## Show the image that is actually being fed to the network
    ## The image was resized while preserving aspect ratio and then
    ## cropped. After that, the mean pixel value was subtracted from
    ## each pixel of that crop. We normalize the image to be between [-1, 1]
    ## to show the image.
    plt.imshow( network_input / (network_input.max() - network_input.min()) )
    plt.suptitle("Resized, Cropped and Mean-Centered input to network",
                 fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        ## Now we print the top-5 predictions that the network gives us with
        ## corresponding probabilities. Pay attention that the index with
        ## class names is shifted by 1 -- this is because some networks
        ## were trained on 1000 classes and others on 1001. VGG-16 was trained
        ## on 1000 classes.
        print('Probability %0.2f => [%s]' % (probabilities[index], names[index+1]))
        
    res = slim.get_model_variables()

# Terminal -
Probability 1.00 => [school bus]
Probability 0.00 => [minibus]
Probability 0.00 => [passenger car, coach, carriage]
Probability 0.00 => [trolleybus, trolley coach, trackless trolley]
Probability 0.00 => [cab, hack, taxi, taxicab]
