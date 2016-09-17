#Tutorial for using caffe pre-trained model

##File descriptions

###Dataset preparation and preprocessing
##### make_datasets.py:
creates two files with the name and location of the training and validation images. The resulting files are required to convert the dataset to the lmdb format
##### create_distracted.sh:
creates the training and validation lmdb files. The word 'distracted' in the name refers to the name of the dataset used here
##### make_driver_mean.sh:
finds the mean image of the training set to be used for normalization by caffe later

### Network definition
##### solver.prototxt:

##### train_val.prototxt:

##### deploy.prototxt:


### Training
#####: train.sh

### Classification
##### classify_driver.py:
##### classify_driver_real.py:
##### classify_driver_batch.py:

### Utility
##### batch_load.py:

##Preparing the dataset

##Getting the pre-trained model and network definition files

##Modifying the files

##Running the training

##Running the prediction
