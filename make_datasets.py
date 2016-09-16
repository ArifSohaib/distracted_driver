import os;
import random;
import math;
#TRAIN_FOLDER = './train/';
CURRENT_FOLDER = os.getcwd();
print(CURRENT_FOLDER)
TRAIN_FOLDER = os.path.join(CURRENT_FOLDER,'train');
print(TRAIN_FOLDER)
folders = os.listdir(TRAIN_FOLDER);
for folder in folders:
    data_folder = os.path.join(TRAIN_FOLDER,folder);
    print(data_folder);
    data = os.listdir(data_folder);

    num_total_images = len(data);
    num_val_images = int(math.ceil(num_total_images * 0.2));
    if(num_val_images%2 != 0):
        num_val_images = num_val_images + 1;
    print('%i validation iamges from %i total' %(num_val_images, num_total_images))
    random.shuffle(data);
    val_images = data[:num_val_images];
    train_images = data[num_val_images+1:];
    with open('val.txt','a') as val:
        for image in val_images:
            i = os.path.join(data_folder,image)
            #val.write(os.path.join(data_folder,image));

            label = i + " " + (str(data_folder[-1])) + "\n";
            val.write(label);
    with open('train.txt','a') as train:
        for image in train_images:
            i = os.path.join(data_folder,image)
            #train.write(os.path.join(data_folder,image));
            label = i + " " + (str(data_folder[-1])) + "\n";
            train.write(label);
