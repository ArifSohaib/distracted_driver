#Tutorial for using caffe pre-trained model

##Brief File descriptions

###Dataset preparation and preprocessing
##### make_datasets.py:
creates two files with the name and location of the training and validation images. The resulting files are required to convert the dataset to the lmdb format
##### create_distracted.sh:
creates the training and validation lmdb files. The word 'distracted' in the name refers to the name of the dataset used here
##### make_driver_mean.sh:
finds the mean image of the training set to be used for normalization by caffe later

### Network definition
##### solver.prototxt:
defines the main network parameters and file locations
##### train_val.prototxt:
defines the network being trained
##### deploy.prototxt:
defines the network for classification(only minor changes from train_val required)

### Training
##### train.sh:
script file to train the network

### Classification
##### classify_driver.py:
using the deploy file on labeled test set to view results and test accuracy on each class
##### classify_driver_real.py:
using the deploy file on unlabeled test data for final results
##### classify_driver_batch.py:
Experimental file to use batch classification. Works but currently running into memory issues.
If you know a fix, please suggest.

### Utility
##### batch_load.py:
Utility file used by the classify_driver_batch.py to load classification images in batch

##Preparing the dataset

##Getting the pre-trained model and network definition files

##Modifying the files

##Running the training

##Running the prediction
