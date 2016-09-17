#creates two txt files with location and category of images
#to use it, put it in the folder that contains the training folder
import os;
import random;
import math;
CURRENT_FOLDER = os.getcwd();
print(CURRENT_FOLDER)
TRAIN_FOLDER = os.path.join(CURRENT_FOLDER,'train');
print(TRAIN_FOLDER)
folders = os.listdir(TRAIN_FOLDER);
#each folder contains a different category so iterate over them all
for folder in folders:
    data_folder = os.path.join(TRAIN_FOLDER,folder);
    print(data_folder);
    data = os.listdir(data_folder);

    num_total_images = len(data);
    #Validation images are made from 20% of the training images
    num_val_images = int(math.ceil(num_total_images * 0.2));
    #number of images in both training and validation sets is kept even
    if(num_val_images%2 != 0):
        num_val_images = num_val_images + 1;
    print('%i validation images from %i total' %(num_val_images, num_total_images))
    #shuffle the data to put randomly selected images in the training and validation set
    random.shuffle(data);
    #put the first few images in the validation set
    val_images = data[:num_val_images];
    #and all the others in the training set
    train_images = data[num_val_images+1:];
    with open('val.txt','a') as val:
        for image in val_images:
            #i is the full path of the image
            i = os.path.join(data_folder,image);
            #in our case, the images are labeled c0 to c9 so the category is the last character of the name which is stored in data_folder
            #differently labeled dataset folders will require a different way to extract labels eg str(data_folder[-2] for 2 digit labels preceded by text
            label = i + " " + (str(data_folder[-1])) + "\n";
            val.write(label);
    #same process as above is repeated with validation images
    with open('train.txt','a') as train:
        for image in train_images:
            i = os.path.join(data_folder,image)
            label = i + " " + (str(data_folder[-1])) + "\n";
            train.write(label);
