###Some Notes:
######NOTE1:
This is my first tutorial so constructive feed-back and questions are highly appreciated and I apologize for any mistakes.
Feel free to contact me at arifsohaib.sa@gmail.com.
######NOTE2:
I am not an expert in caffe but this project was successfully completed by my as my first kaggle entry and I found caffe tutorials hard to come by so I thought I would share.
######NOTE3:
I used various sources from the Internet to learn caffe and fix issues I faced so some code is modified from other sources which I did not keep track of.
I apologize for not crediting them. If you find such snippits, please inform me and I will add them.

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
Be warned that this file could crash your system due to the memory issues.
If you know a fix, please suggest.
##### classify_driver_batch_testing.py:
Experimental file that uses the batch loader on a much smaller subset and thus does not crash and verifies that the loader works.

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
The file has been annotated to be usable for tutorial purposes.

Once we have these txt files, we can use the file create_distracted.sh to create the lmdb files (side-note:I know it could be named better but I could not think of anything else at the time).

After creating the lmdb files, we need the mean image. This can be created using the make_driver_mean.sh which runs a command line tool called 'compute_image_mean' on the training files and stores the result in a binaryproto file.

##Getting the pre-trained model and network definition files
In most cases, training a large network with little training data will lead to over-fitting on the training set and low accuracy in validation and testing.
To solve this issue, a pre-trained network can be used.
This network has already learned the lower level features in its earlier layers. For example it 'knows' what a dots and shapes look like in the lowest layers and in higher ones, it 'knows' how they are combined in an image. Its only in the highest layer that it actually does classification. So by 'freezing' the lower layers, the network retains information about what an image looks like but the highest layers can be retrained to actually recognize/categorize the image.

To do this, we need to get 4 files and modify 3 of them. The files depend on the network you are training.
In my case, it was Caffenet, a modified version of Alexnet, that comes with caffe.
The files required can be found in the caffe folder under the models folder as bvlc_reference_caffenet.
You can find solver.prototxt, train_val.prototxt and deploy.prototxt here.
The trained model can be found in the description here: https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet
It is separate from the main repo as the size is too large.

##Modifying the files
You need to modify the model to point to your data and change the number of classifiers as well as disable training on some of the earlier nodes.
#####solver.prototxt:
If you stored your model somewhere else or changed the filename, then you need to mention the location in the solver.prototxt file.
The 'net' parameter defines the file where your network is stored. Originally it will be something like this:\n
net: "models/bvlc_alexnet/train_val.prototxt"\n
Note that the location is defined relative to the caffe root folder.
In my case, the model was in a folder called mymodeldef in the caffe folder so I changed it to this:\n
net: "mymodeldef/train_val.prototxt"\n
The 'test_iter' parameter defines the number of forward passes to run with the batches of validation images, while the batch_size is defined in the train_val file, I will explain how to do that change in the next section. These should be defined in a way that covers the whole validation set.\n
For example in my case, the validation set consisted of 4494 images so I defined the test_iter as 107 and batch_size in test_val.prototxt as 42 so 107x42=4494.
The numbers were found by finding the factors of the size of the validation set. This is the reason I didn't take exactly 20% of training images as validation set.\n
'test_interval' defines the number of iterations after which the training temporarily stops and the model is checked against the validation set instead. The accuracy on the validation set is displayed after this number of iterations.\n
'base_lr' is the base learning rate for all layers. You can define a different learning rate for a layer in the train_val file and that will be multiplied by the base_lr.\n
'lr_policy' defines how the base_lr is changed. Possible values and their description is defined here https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L157-L172
In my case, I used the default 'step' policy which returns the learning rate as base_lr * gamma ^ (floor(iter / step)) \n
'gamma' a value indicating how much the learning rate should change. It is used slightly differently for each lr_policy.\n
'stepsize' a value indicating how often the learning rate should be decreased.\n
'display' is the number of iterations after which the loss of the model is displayed\n
'max_iter' is the maximum number of iterations that the model should run for. It can be manually stopped earlier if needed.\n
'momentum' the weight of the previous update. Full details here: http://caffe.berkeleyvision.org/tutorial/solver.html (also contains details on the other parameters)\n
'weight_decay' parameter to control the effect of regularization added to the weights. If the number of examples you have is low, reduce this term. For more details see http://stackoverflow.com/questions/32177764/what-is-weight-decay-meta-parameter-in-caffe \n
'snapshot' is the number of iterations after which the model is saved in a caffemodel file. The saved snapshot can be used to continue training or fine-tuning\n
'snapshot_prefix' defines the relative location(relative to caffe root folder) and file prefix of the solver file. Note that if the folder in the location does not exits, you will have to manually create it.\n
'solver_mode' specifies if the model should run on the CPU or GPU. \n
#####train_val.prototxt:
This file defines the network.
In our case, we are retraining the networks's last layers so we need to disable the learning on the others.
Additionally, we need to point to our data source(the lmdb file created in the last section) and change the final classification layer to only output the number of categories we have.
Optionally, we need to change the location of the mean_image.

The layers are defined in a JSON like syntax.
The first 2 layers are data layers.

Within the top layers there is a section called transrofm_param which defines any transformation applied to the image.
Here there is a parameter called mean_image. This needs to point to the relative location of your mean image file(again, relative to the caffe directory).
In my case, the mean image was created in mydata/distracted_driver/ by the script make_driver_mean.sh by the name driver_mean.binaryproto so the parameter is "mydata/distracted_driver/driver_mean.binaryproto".

The parameter 'top' in the layer defines the layer being output.
In later layers, the parameter 'bottom' defines the name of the layer where the given layer gets its input.
Layers that don't have any parameters, like relu, should have the same name for the top and bottom parameters.

The phase section defines weather to use this data layer in training(TRAIN) or validation(TEST).

The data_param defines the location of the data. You need to change it to the relative location of the LMDB file of the training data, in case of TRAIN phase.
and validation data in case of the TEST phase.
They both can also have different batch sizes and as mentioned in the previous section, the batch_size of the TEST phase needs to be equal to the size of the validation set divided by test_iter from the solver file.
The backend defines which type of data is being used.

After this, we need to set lr_mult: 0 (this is multiplied by the base_lr and gamma) for every layer that we don't want to train. The classification layer(the last layer) and the layer before that was retrained in my case to I kept lr_mult: 1 for them.

In the classification layer(fc8 in my case) we need to set num_output to the number of categories we need to classify.

With these changes we are ready to re-train the network on our data/categories.

#####deploy.prototxt:
The deploy.prototxt can be made by simply copying your train_val.prototxt and making a few changes. This is actually one of caffe's weaknesses as programmers used to object oriented code would want to simplify this by using just a transformation of train_val instead of copying it.\n
To convert train_val to deploy, we need to change the data layer. In training we knew exactly where the images were coming from, however, in testing and deployment we don't moreover, in training we also had a validation phase called 'test' which is not needed in the deployment. So the data layer is changed to reflect these changes.
Remove your entire data layer and replace it with something like this:\n
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape: { dim: 10 dim: 3 dim: 227 dim: 227 } }
}\n
The dim in the shape of the input_param represnt the number of images, then the color channels, followed by the height and then the width.
Don't worry if your test images will have different dimensions, that will be fixed in preprocessing in the classification script.

Next we need to change the remove the accuracy(type:Accuracy) and loss(type:SoftmaxWithLoss) layers from the train_val and insert a softmax layer. The reason for this is that during training and validation, we had labels to test our loss and accuracy while in deployment, we have unlabeled data.
You can verify this by seeing that both accuracy and loss layers take as input("bottom") the final classification layer(in our case fc8) and the labels and in deployment we don't have the labels.

##Running the training
Running the training is as simple as running one line on the terminal as you can see in the code of train.sh.\n
./build/tools/caffe train -solver mymodeldef/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel\n
The solver parameter here needs to point to your solver file,
The weights parameter points to the pre-trained model you downloaded in the beginning and can be left blank if you want to train from scratch or replaced with one of your snapshots if you want to resume training.
Note that this command needs to run from the caffe root directory.

##Running the prediction
If your test set is large, a small step you should do before running the classifier on the full set is to take a small subset of it and label it.
To properly test for bias, you should use about the same number of images from each category. For example, in my case when I trained the classifier from scratch, the resulting model was highly biased to one of the classes, the class with the fewest training examples but I mistook it for a good classifier as I didn't test it on other classes until later. Note, however, that you might need to manually label this subset unless your training set is large enough that you can set aside a test set in addition to the training and validation sets.

That said, in my case, you will notice that there are 3 files for prediction(classify_driver.py, classify_driver_real.py classify_driver_batch_testing.py and classify_driver_batch.py). All of these use Caffe's python interface so you should make sure to install it when installing caffe, if you have trouble with this you can install Nvidia DIGITS which also installs pycaffe.

The classify_driver.py file is the one that is used to test the model on unseen labeled test images and produces a csv file.
The classify_driver_real.py classifies each unlabeled image in the test set one by one and produces a csv file.
Both files have been annotated to be easy to modify and use.
The classify_driver_batch.py and classify_driver_batch_testing.py are experimental files to test the batch loader. They are similar to the other two but have not been annotated.
