import os;
import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import gc

import caffe;
class batch_load:
    def __init__(self, folder,batch_size):
        self.used_files = []
        self.total_size = len(os.listdir(folder))
        self.folder = folder
        self.batch_size = batch_size

    def batch_load(self, verbose = False):
#        print(folder)
        image_batch = []
        batch_file_names = []
        for image in os.listdir(self.folder):
            IMAGE_FILE = os.path.join(self.folder, image)
            self.used_files.append(IMAGE_FILE)
            #print(used_files)
            if (len(image_batch) < min(self.batch_size, self.total_size)):
                if(verbose):
                    print "adding ", IMAGE_FILE
                if(image not in self.used_files):
                    input_image = caffe.io.load_image(IMAGE_FILE)
                    image_batch.append(input_image)
                    batch_file_names.append(image)
                else:
                    print("file already in batch")
                    continue
            else:
                break
        return batch_file_names, image_batch;

    def test(self):
        FOLDER = '/home/sohaib/kaggle/test'
        batch = self.batch_load(FOLDER,50)

class batch_load_testing:
    def __init__(self,folder,batch_size):
        self.folder = folder;
        self.batch_size = batch_size;

    def build_batch(self):
        #list all the files in the folder
        files = os.listdir(self.folder);
        batch = [];
        #diveide the file names into batches of lists
        #batch[0] = files[0:N]
        #batch[1] = files[N+1:N+N]
        #batch[2] = files[N+N+N+1:N+N+N+N]
        for i in range(0,len(files)/self.batch_size, self.batch_size):
            batch.append(files[i:i+self.batch_size])
            del files[i:i+self.batch_size]
            gc.collect()
        batch.append(files)#put all the remaining files in the last batch
        return batch;

    def get_one_batch(self,batch):
        image_batch = [];
        image_batch_names = [];
        for image in batch:
            IMAGE_FILE = os.path.join(self.folder, image)
            image_batch.append(caffe.io.load_image(IMAGE_FILE));
            image_batch_names.append(image); #this list of image names is seperate from the batch to allow both the lists to have the same order
            gc.collect()
        return image_batch_names, image_batch;
