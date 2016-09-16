#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in mydata

EXAMPLE=../mydata/distracted_driver

DATA=../mydata/distracted_driver
TOOLS=../build/tools

$TOOLS/compute_image_mean $EXAMPLE/train \
  $DATA/driver_mean.binaryproto

echo "Done."
