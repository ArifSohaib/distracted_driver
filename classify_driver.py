import numpy as np
import matplotlib.pyplot as plt
import os
from batch_load import batch_load
# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../mymodeldef/deploy.prototxt' #The deploy file
PRETRAINED = '../mymodeldef/snapshots/driver_train1_iter_30000.caffemodel' #the model you trained
IMAGE_ROOT = '/home/sohaib/kaggle/test_labeled/' #The labeled test images(seperate from training and validation)
IMAGE_CATS = os.listdir(IMAGE_ROOT)

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1), #The file mentioned here is the mean of imagenet, you can also use your mean image after converting it to .npy
channel_swap=(2,1,0),
raw_scale=255,
image_dims=(256, 256))

#a result file is being appended to later, to prevent adding the same image multiple times, check if the file is preasent
assert((os.path.isfile('result.csv'))) == False

#The labeled test images are in different folders based on their category so we run the classifier for each category folder
for cat in IMAGE_CATS:
    SUB_PATH = os.path.join(IMAGE_ROOT,cat)
    print(SUB_PATH)
    #since the batch loader does work on small subsets of data, we can safely use it to speed up classification instead of doing it one by one
    load = batch_load(SUB_PATH,50);
    file_names, input_images = load.batch_load()
    prediction = net.predict(input_images)  # predict takes any number of images, and formats them for the Caffe net automatically

    for i in range(len(prediction)):
        print 'image: ', file_names[i], 'real class: ', cat, ' predicted class:', prediction[i].argmax()
        with open('result.csv','a') as result:
            result.write("%s,"%file_names[i])
            for pred in prediction[i]:
                result.write("%.3f,"%pred)
            result.write("\n")
