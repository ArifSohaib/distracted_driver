import numpy as np
import matplotlib.pyplot as plt
import os
# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/sohaib/caffe/mymodeldef/deploy.prototxt' #The deploy file
PRETRAINED = '/home/sohaib/caffe/mymodeldef/snapshots/driver_train1_iter_30000.caffemodel' #The trained model
IMAGE_ROOT = '/home/sohaib/kaggle/test/' #unlabeled test images

#since we append to the result file later, we need to check if it already exists to avoid adding the same image multiple times
assert((os.path.isfile('result_real.csv'))) == False
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
    mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), #The file mentioned here is the mean of imagenet, you can also use your mean image after converting it to .npy
    channel_swap=(2,1,0), #The color channels are different in the main caffe C++ implementation and the python interface https://skyuuka.wordpress.com/2015/04/14/about-channel-swap-in-caffe/
    raw_scale=255,
    image_dims=(256, 256))
#All the test images are in the same folder so we read all of them
for img in os.listdir(IMAGE_ROOT):
    IMAGE_FILE = os.path.join(IMAGE_ROOT,img)
    print(IMAGE_FILE)
    input_image = caffe.io.load_image(IMAGE_FILE)

    #Since the batch loader runs into memory issues we load only one image at a time
    prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
    with open('result_real.csv','a') as result:
        result.write("%s,"%img)
        for pred in prediction[0]:
            result.write("%.3f,"%pred)
        result.write("\n")
