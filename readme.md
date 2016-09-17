###Some Notes:
######NOTE1:
This is my first tutorial so constructive feed-back and questions are highly appreciated and I apologize for any mistakes.
######NOTE2:
I am not an expert in caffe but this project was successfully completed by my as my first kaggle entry and I found caffe tutorials hard to come by so I thought I would share.

#Tutorial for using caffe pre-trained model

##Brief File descriptions

####Dataset preparation and preprocessing
##### make_datasets.py:
creates two files with the name and location of the training and validation images. The resulting files are required to convert the dataset to the lmdb format
##### create_distracted.sh:
creates the training and validation lmdb files. The word 'distracted' in the name refers to the name of the dataset used here
##### make_driver_mean.sh:
finds the mean image of the training set to be used for normalization by caffe later

####Network definition
##### solver.prototxt:
defines the main network parameters and file locations
##### train_val.prototxt:
defines the network being trained
##### deploy.prototxt:
defines the network for classification(only minor changes from train_val required)

####Training
##### train.sh:
script file to train the network

####Classification
##### classify_driver.py:
using the deploy file on labeled test set to view results and test accuracy on each class
##### classify_driver_real.py:
using the deploy file on unlabeled test data for final results
##### classify_driver_batch.py:
Experimental file to use batch classification. Works but currently running into memory issues.
If you know a fix, please suggest.

####Utility
##### batch_load.py:
Utility file used by the classify_driver_batch.py to load classification images in batch

##Preparing the dataset:
You first need a dataset of images as caffe is primarily designed to work with image data.
In my case I started using caffe as part of the following kaggle competition: https://www.kaggle.com/c/state-farm-distracted-driver-detection/ and thus used its data.
In this case, the data was a large number of labeled and unlabeled images.
The labeling was done by putting the images for each label in folders named for their label. eg: category 0 (normal driving) images were in a folder called c0 and so on for each of the 10 category. These labeled images were in a folder called 'train'.
In addition to these, there were unlabeled images whose class probabilities were to be reported. These were all in a folder called 'test'
There was no validation set, so I created it from about 20% of the training set.

All images were in jpg format and can be viewed directly however, caffe does not directly use these images for training.
For training I decided to put the images in lmdb format as the alternative was hdf5 and I was running into memory issues with it and lmdb is also the most used format with caffe.

To prepare the lmdb file you need two text files with the location and category of the dataset images. Manually making these files for large datasets is almost impossible so I made a python script to do this for me. This script is the make_datasets.py file.
The file has been commented to be usable for tutorial purposes.

Once we have these txt files, we can use the file create_distracted.sh to create the lmdb files (side-note:I know it could be named better but I could not think of anything else at the time).

After creating the lmdb files, we need the mean image. This can be created using the make_driver_mean.sh which runs a command line tool called 'compute_image_mean' on the training files and stores the result in a binaryproto file.

##Getting the pre-trained model and network definition files
In most cases, training a large network with little training data will lead to over-fitting on the training set and low accuracy in validation and testing.
To solve this issue, a pre-trained network can be used.
This network has already learned the lower level features in its earlier layers. For example it 'knows' what a dots and shapes look like in the lowest layers and in higher ones, it 'knows' how they are combined in an image. Its only in the highest layer that it actually does classification. So by 'freezing' the lower layers, the network retains information about what an image looks like but the highest layers can be retrained to actually recognize/categorize the image.

To do this, we need to get 4 files and modify 3 of them. The files depend on the network you are training.
In my case, it was Alexnet that comes with caffe.
The files required can be found in the caffe folder under the models folder.
You can find solver.prototxt, train_val.prototxt and deploy.prototxt here.
The trained model can be found here: https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet

##Modifying the files

##Running the training

##Running the prediction
