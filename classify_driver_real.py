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
MODEL_FILE = '/home/sohaib/caffe/mymodeldef/deploy.prototxt'
PRETRAINED = '/home/sohaib/caffe/mymodeldef/snapshots/driver_train1_iter_20000.caffemodel'
IMAGE_ROOT = '/home/sohaib/kaggle/test/'


assert((os.path.isfile('result_real.csv'))) == False
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
    mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
    channel_swap=(2,1,0),
    raw_scale=255,
    image_dims=(256, 256))

for img in os.listdir(IMAGE_ROOT):
    IMAGE_FILE = os.path.join(IMAGE_ROOT,img)
    print(IMAGE_FILE)
    # caffe.set_mode_cpu()

    input_image = caffe.io.load_image(IMAGE_FILE)
    # plt.imshow(input_image)

    prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
    # print 'prediction shape:', prediction[0].shape ##DEBUGGING CODE

    # plt.plot(prediction[0]) 
    #print 'predicted class:', prediction[0].argmax()
    # plt.show()
    with open('result_real.csv','a') as result:
        #result_string = str(img) + "," + str(prediction[0]) + '\n';
        result.write("%s,"%img)
        for pred in prediction[0]:
            result.write("%.3f,"%pred)
        result.write("\n")
