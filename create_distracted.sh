#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

#Some setup defining the folders.
#Replace with the path to the folder where you want you lmdb files
OUTPUT_FOLDER=/home/sohaib/caffe/mydata/distracted_driver_orig
#Replace with the path to the folder that contains your train folder
DATA=/home/sohaib/kaggle
#Replace with the location of your caffe tools directory, which will be in your caffe directory. i.e /path/to/caffe/build/tools
TOOLS=/home/sohaib/caffe/build/tools

#if your validation or training data is somewhere else in the DATA folder, then add the additional path here
TRAIN_DATA_ROOT=/
VAL_DATA_ROOT=/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

#Check if data directory exists
if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi
#check if output folder exists
if [ ! -d "$OUTPUT_FOLDER" ]; then
  echo "Creating output folder: $OUTPUT_FOLDER"
  mkdir $OUTPUT_FOLDER
fi

#actual script to use the convert_imageset tool to make the lmdb files
echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $OUTPUT_FOLDER/train

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $OUTPUT_FOLDER/val

echo "Done."
