#!/bin/bash

FMSG="- Acute Lymphoblastic Leukemia Tensorflow CNN For UP2 installation terminated"

read -p "? This script will install the Acute Lymphoblastic Leukemia Tensorflow CNN For UP2 required Python libraries and Tensorflow on your device. Are you ready (y/n)? " cmsg

if [ "$cmsg" = "Y" -o "$cmsg" = "y" ]; then

    echo "- Installing required Python libraries and Tensorflow"

    sudo apt update
    sudo apt -y install cmake

    pip3 install --user scikit-build
    pip3 install --user opencv-python
    pip3 install --user geocoder
    pip3 install --user imutils
    pip3 install --user jsonpickle
    pip3 install --user paho-mqtt
    pip3 install --user psutil
    pip3 install --user tensorflow

else
    echo $FMSG;
    exit
fi