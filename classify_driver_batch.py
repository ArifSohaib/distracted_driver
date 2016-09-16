import numpy as np
import matplotlib.pyplot as plt
import os
from batch_load import batch_load_testing
import gc

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../mymodeldef/deploy.prototxt'
PRETRAINED = '../mymodeldef/snapshots/driver_train1_iter_30000.caffemodel'
IMAGE_ROOT = '/home/sohaib/kaggle/test/'
BATCH_SIZE = 64
caffe.set_mode_gpu()
assert((os.path.isfile('result_real.csv'))) == False
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
    mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
    channel_swap=(2,1,0),
    raw_scale=255,
    image_dims=(256, 256))
#TODO: Check if images are already classified
#images in result_real.csv are already classified
#find images in result_real.csv and remove them from IMAGE_ROOT

load = batch_load_testing(IMAGE_ROOT,BATCH_SIZE)
batches = load.build_batch()
batch_count = 0;
for batch in batches:
    print ("classifying batch %i" %batch_count)
    batch_file_names, batch_images = load.get_one_batch(batch)
    prediction = net.predict(batch_images)
    batch_images = []  #clear the images to save memory
    gc.collect()
    #CONTINUE HERE
    for i in range(len(prediction)):
        with open('result_real.csv','a') as result:
            result.write("%s,"%batch_file_names[i])
            for pred in prediction[i]:
                result.write("%f,"%pred)
            result.write("\n")
    #done with the batch so clear the image names
    batch_file_names = []
    batch_count += 1;
    gc.collect()

# print len(os.listdir(IMAGE_ROOT))
# for i in range(len(os.listdir(IMAGE_ROOT))/BATCH_SIZE,BATCH_SIZE):
#     batch_file_names[i], batch[i] = load.batch_load(IMAGE_ROOT,50)
#     prediction[i] = net.predict(batch[i])
#     for pred in range(prediction[i]):
#         print "image:", batch_file_names[i][pred], " predicted class: ", prediction[i][pred]
#
# for img in os.listdir(IMAGE_ROOT):
#     IMAGE_FILE = os.path.join(IMAGE_ROOT,img)
#     print(IMAGE_FILE)
#     # caffe.set_mode_cpu()
#
#     input_image = caffe.io.load_image(IMAGE_FILE)
#     # plt.imshow(input_image)
#
#     prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
#     # print 'prediction shape:', prediction[0].shape
#     # plt.plot(prediction[0])
#     #print 'predicted class:', prediction[0].argmax()
#     # plt.show()
#     with open('result_real.csv','a') as result:
#         #result_string = str(img) + "," + str(prediction[0]) + '\n';
#         result.write("%s,"%img)
#         for pred in prediction[0]:
#             result.write("%.3f,"%pred)
#         result.write("\n")
