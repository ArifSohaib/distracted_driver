import numpy as np
import matplotlib.pyplot as plt
import os
from batch_load import batch_load_testing
# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../mymodeldef/deploy.prototxt'
PRETRAINED = '../mymodeldef/snapshots/driver_train1_iter_30000.caffemodel'
IMAGE_ROOT = '/home/sohaib/kaggle/test_labeled/'
IMAGE_CATS = os.listdir(IMAGE_ROOT)

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
channel_swap=(2,1,0),
raw_scale=255,
image_dims=(256, 256))

assert((os.path.isfile('result.csv'))) == False
for cat in IMAGE_CATS:
    SUB_PATH = os.path.join(IMAGE_ROOT,cat)
    print(SUB_PATH)
    batch_obj = batch_load_testing(SUB_PATH,5)#There should be 2 batches for each every category
    batches = batch_obj.build_batch()
    for batch in batches:
        file_names, input_images = batch_obj.get_one_batch(batch)
        prediction = net.predict(input_images)  # predict takes any number of images, and formats them for the Caffe net automatically

        for i in range(len(prediction)):
            print 'image: ', file_names[i], 'real class: ', cat, ' predicted class:', prediction[i].argmax()
            with open('result.csv','a') as result:
                result.write("%s,"%file_names[i])
                for pred in prediction[i]:
                    result.write("%.3f,"%pred)
                result.write("\n")
