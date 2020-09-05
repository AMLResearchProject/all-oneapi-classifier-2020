# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
## OneAPI Acute Lymphoblastic Leukemia Classifier
### OneAPI OpenVINO Raspberry Pi 4 Acute Lymphoblastic Leukemia Classifier

![OneAPI Acute Lymphoblastic Leukemia Classifier](../Media/Images/Peter-Moss-Acute-Myeloid-Lymphoblastic-Leukemia-Research-Project.png)

&nbsp;

# Table Of Contents

- [Introduction](#introduction)
- [DISCLAIMER](#disclaimer)
- [ALL-IDB](#all-idb)
  - [ALL_IDB1](#all_idb1)
- [Network Architecture](#network-architecture)
  - [Results Overview](#results-overview)
- [Installation](#installation)
- [Getting Started](#getting-started)
    - [Data](#data)
    - [Configuration](#configuration)
- [Local Testing (CPU)](#local-testing-cpu)
    - [Local Testing Results (CPU)](#local-testing-results-cpu)
- [Server Testing (CPU)](#server-testing-cpu)
    - [Server Testing Results (CPU)](#server-testing-results-cpu)
- [Local Testing (OpenVINO/NCS2)](#local-testing-openvinoncs2)
    - [Local Testing Results (CPU)](#local-testing-results-openvinoncs2)
- [Server Testing (OpenVINO/NCS2)](#server-testing-openvinoncs2)
    - [Server Testing Results (OpenVINO/NCS2)](#server-testing-results-openvinoncs2)
- [Contributing](#contributing)
  - [Contributors](#contributors)
- [Versioning](#versioning)
- [License](#license)
- [Bugs/Issues](#bugs-issues)

&nbsp;

# Introduction
In this project you will use the model trained in the [OneAPI Acute Lymphoblastic Leukemia Classifier CNN](../CNN) project to create an [Intermediate Representation (IR)](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_IR_and_opsets.html) which allow the model to be used with the [Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html).

You will deploy the model to a Raspberry Pi 4 and test the classifier running on CPU and with OpenVINO using the [Intel® Movidius™ Neural Compute Stick 2](https://software.intel.com/content/www/us/en/develop/hardware/neural-compute-stick.html), benchmarking the results.

&nbsp;

# DISCLAIMER
These projects should be used for research purposes only. The purpose of the projects is to show the potential of Artificial Intelligence for medical support systems such as diagnosis systems.

Although the classifiers are accurate and show good results both on paper and in real world testing, they are not meant to be an alternative to professional medical diagnosis.

Developers that have contributed to this repository have experience in using Artificial Intelligence for detecting certain types of cancer. They are not doctors, medical or cancer experts.

Please use this system responsibly.

&nbsp;

# ALL-IDB
You need to be granted access to use the Acute Lymphoblastic Leukemia Image Database for Image Processing dataset. You can find the application form and information about getting access to the dataset on [this page](https://homes.di.unimi.it/scotti/all/#download) as well as information on how to contribute back to the project [here](https://homes.di.unimi.it/scotti/all/results.php). If you are not able to obtain a copy of the dataset please feel free to try this tutorial on your own dataset, we would be very happy to find additional AML & ALL datasets.

## ALL_IDB1
In this project, [ALL-IDB1](https://homes.di.unimi.it/scotti/all/#datasets) is used, one of the datsets from the Acute Lymphoblastic Leukemia Image Database for Image Processing dataset.

"The ALL_IDB1 version 1.0 can be used both for testing segmentation capability of algorithms, as well as the classification systems and image preprocessing methods. This dataset is composed of 108 images collected during September, 2005. It contains about 39000 blood elements, where the lymphocytes has been labeled by expert oncologists. The images are taken with different magnifications of the microscope ranging from 300 to 500."

&nbsp;

# Network Architecture
<img src="https://www.leukemiaresearchfoundation.ai/github/media/images/paper_1_architecture.png" alt="Proposed Architecture" />

_Fig 1. Proposed architecture ([Source](https://airccj.org/CSCP/vol7/csit77505.pdf "Source"))_

In [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System"), the authors propose a simple 5 layer Convolutional Neural Network.

In the OneAPI Acute Lymphoblastic Leukemia Classifier CNN project you used an augmented dataset with the network proposed in this paper. You built a Convolutional Neural Network, as shown in Fig 1, consisting of the following 5 layers (missing out the zero padding layers). Note you used conv sizes of (100x100x30) whereas in the paper, the authors use (50x50x30).

- Conv layer (100x100x30)
- Conv layer (100x100x30)
- Max-Pooling layer (50x50x30)
- Fully Connected layer (2 neurons)
- Softmax layer (Output 2)

&nbsp;

# Installation
Follow the [installation guide](Documentation/Installation.md) to install the requirements for this project.

&nbsp;

# Getting Started

## Data
Once you have your data you need to add it to the project filesystem. You will notice the data folder in the Model directory, **Model/Data**, inside you have **Test**.

You need to use the following test data if you are using the pre-trained model, or use the same test data you used when training your own network if you used a different test set there.

- im006_1.jpg
- im020_1.jpg
- im024_1.jpg
- im026_1.jpg
- im028_1.jpg
- im031_1.jpg
- im035_0.jpg
- im041_0.jpg
- im047_0.jpg
- im053_1.jpg
- im057_1.jpg
- im060_1.jpg
- im063_1.jpg
- im069_0.jpg
- im074_0.jpg
- im088_0.jpg
- im095_0.jpg
- im099_0.jpg
- im101_0.jpg
- im106_0.jpg

&nbsp;

## Configuration

[config.json](Model/config.json "config.json")  holds the configuration for our network. You need to update the **cnn->system->server** field with the IP of your Raspberry Pi 4, you should also change the port.

```
{
    "cnn": {
        "system": {
            "cores": 8,
            "server": "",
            "port": 1234
        },
        "core": [
            "Server",
            "Client",
            "Classify"
        ],
        "data": {
            "dim": 100,
            "file_type": ".jpg",
            "labels": [0, 1],
            "test": "Model/Data/Test",
            "test_data": [
                "im006_1.jpg",
                "im020_1.jpg",
                "im024_1.jpg",
                "im026_1.jpg",
                "im028_1.jpg",
                "im031_1.jpg",
                "im035_0.jpg",
                "im041_0.jpg",
                "im047_0.jpg",
                "im053_1.jpg",
                "im057_1.jpg",
                "im060_1.jpg",
                "im063_1.jpg",
                "im069_0.jpg",
                "im074_0.jpg",
                "im088_0.jpg",
                "im095_0.jpg",
                "im099_0.jpg",
                "im101_0.jpg",
                "im106_0.jpg"
            ],
            "valid_types": [
                ".JPG",
                ".JPEG",
                ".PNG",
                ".GIF",
                ".jpg",
                ".jpeg",
                ".png",
                ".gif"
            ]
        },
        "model": {
            "device": "CPU",
            "freezing_log_dir": "Model/Freezing",
            "frozen": "frozen.pb",
            "ir": "Model/IR/frozen.xml",
            "model": "Model/model.json",
            "saved_model_dir": "Model",
            "weights": "Model/weights.h5"
        },
        "rpi4": {
            "freezing_log_dir": "Model/Freezing",
            "frozen": "frozen.pb",
            "ir": "Model/IR/frozen.xml",
            "inScaleFactor": 0.007843,
            "meanVal": 0
        }
    }
}
```

The cnn object contains 3 Json Objects (api, data and model) and a JSON Array (core). **api** has the information used to set up your server you will need to add your local ip, **data** has the configuration related to preparing the training and validation data, and **model** holds the model file paths.

&nbsp;

# Local Testing (CPU)
Now we will use the test data to see how the classifier reacts to our testing data on a Raspberry Pi 4. Real world testing is the most important testing, as it allows you to see the how the model performs in a real world environment.

This part of the tutorial tests the CNN model on the Raspberry Pi 4 and the test data from the **Model/Data/Test** directory, using the Raspbery Pi 4 CPU. The command to start testing is as follows:

```
 python ALLoneAPI.py Classify
```

## Local Testing Results (CPU)

```
2020-09-05 06:56:11,169 - Core - INFO - Class initialization complete.
2020-09-05 06:56:11,171 - Model - INFO - Class initialization complete.
2020-09-05 06:56:11,171 - Core - INFO - ALLoneAPI RPI4 CNN initialization complete.
2020-09-05 06:56:11,371 - Model - INFO - Model loaded
Model: "AllDS2020_TF_CNN"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
zero_padding2d (ZeroPadding2 (None, 104, 104, 3)       0
_________________________________________________________________
conv2d (Conv2D)              (None, 100, 100, 30)      2280
_________________________________________________________________
zero_padding2d_1 (ZeroPaddin (None, 104, 104, 30)      0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 100, 100, 30)      22530
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 50, 50, 30)        0
_________________________________________________________________
flatten (Flatten)            (None, 75000)             0
_________________________________________________________________
dense (Dense)                (None, 2)                 150002
_________________________________________________________________
activation (Activation)      (None, 2)                 0
=================================================================
Total params: 174,812
Trainable params: 174,812
Non-trainable params: 0
_________________________________________________________________
2020-09-05 06:56:11,581 - Model - INFO - Loaded test image Model/Data/Test/Im028_1.jpg
2020-09-05 06:56:11,917 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.5426948070526123 seconds.
2020-09-05 06:56:12,386 - Model - INFO - Loaded test image Model/Data/Test/Im060_1.jpg
2020-09-05 06:56:12,557 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.639235258102417 seconds.
2020-09-05 06:56:13,026 - Model - INFO - Loaded test image Model/Data/Test/Im057_1.jpg
2020-09-05 06:56:13,195 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.6378085613250732 seconds.
2020-09-05 06:56:13,664 - Model - INFO - Loaded test image Model/Data/Test/Im041_0.jpg
2020-09-05 06:56:13,832 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.6365838050842285 seconds.
2020-09-05 06:56:14,301 - Model - INFO - Loaded test image Model/Data/Test/Im106_0.jpg
2020-09-05 06:56:14,471 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.6382412910461426 seconds.
2020-09-05 06:56:14,939 - Model - INFO - Loaded test image Model/Data/Test/Im101_0.jpg
2020-09-05 06:56:15,114 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.6430854797363281 seconds.
2020-09-05 06:56:15,585 - Model - INFO - Loaded test image Model/Data/Test/Im088_0.jpg
2020-09-05 06:56:15,753 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive) in 0.6380867958068848 seconds.
2020-09-05 06:56:15,960 - Model - INFO - Loaded test image Model/Data/Test/Im026_1.jpg
2020-09-05 06:56:16,124 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.3704390525817871 seconds.
2020-09-05 06:56:16,331 - Model - INFO - Loaded test image Model/Data/Test/Im031_1.jpg
2020-09-05 06:56:16,496 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.3716132640838623 seconds.
2020-09-05 06:56:16,704 - Model - INFO - Loaded test image Model/Data/Test/Im024_1.jpg
2020-09-05 06:56:16,882 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.38602185249328613 seconds.
2020-09-05 06:56:17,352 - Model - INFO - Loaded test image Model/Data/Test/Im099_0.jpg
2020-09-05 06:56:17,522 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.6391148567199707 seconds.
2020-09-05 06:56:17,729 - Model - INFO - Loaded test image Model/Data/Test/Im020_1.jpg
2020-09-05 06:56:17,894 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.37194228172302246 seconds.
2020-09-05 06:56:18,365 - Model - INFO - Loaded test image Model/Data/Test/Im047_0.jpg
2020-09-05 06:56:18,534 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.6396605968475342 seconds.
2020-09-05 06:56:19,006 - Model - INFO - Loaded test image Model/Data/Test/Im053_1.jpg
2020-09-05 06:56:19,182 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly not detected (False Negative) in 0.6474871635437012 seconds.
2020-09-05 06:56:19,390 - Model - INFO - Loaded test image Model/Data/Test/Im006_1.jpg
2020-09-05 06:56:19,558 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.3754298686981201 seconds.
2020-09-05 06:56:20,027 - Model - INFO - Loaded test image Model/Data/Test/Im074_0.jpg
2020-09-05 06:56:20,195 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.6361985206604004 seconds.
2020-09-05 06:56:20,666 - Model - INFO - Loaded test image Model/Data/Test/Im069_0.jpg
2020-09-05 06:56:20,835 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.6396892070770264 seconds.
2020-09-05 06:56:21,306 - Model - INFO - Loaded test image Model/Data/Test/Im063_1.jpg
2020-09-05 06:56:21,482 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.646582841873169 seconds.
2020-09-05 06:56:21,952 - Model - INFO - Loaded test image Model/Data/Test/Im035_0.jpg
2020-09-05 06:56:22,122 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.6393959522247314 seconds.
2020-09-05 06:56:22,567 - Model - INFO - Loaded test image Model/Data/Test/Im095_0.jpg
2020-09-05 06:56:22,736 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive) in 0.6139969825744629 seconds.
2020-09-05 06:56:22,737 - Model - INFO - Images Classifier: 20
2020-09-05 06:56:22,737 - Model - INFO - True Positives: 9
2020-09-05 06:56:22,737 - Model - INFO - False Positives: 2
2020-09-05 06:56:22,737 - Model - INFO - True Negatives: 8
2020-09-05 06:56:22,738 - Model - INFO - False Negatives: 1
2020-09-05 06:56:22,738 - Model - INFO - Total Time Taken: 11.35330843925476
```

We see that the testing is a lot slower than when you tested your model on the Windows machine, the accuracy remains the same.

&nbsp;

# Server Testing (CPU)
Now we will use the test data to see how the server classifier reacts using the CNN model on the Raspberry Pi CPU.

You need to open two terminal windows or tabs, in the first, use the following command to start the server:

```
python ALLoneAPI.py Server
```

In your second terminal, use the following command:

```
python ALLoneAPI.py Client
```

## Server Testing Results (CPU)

```
2020-09-05 07:03:40,725 - Core - INFO - Class initialization complete.
2020-09-05 07:03:40,727 - Model - INFO - Class initialization complete.
2020-09-05 07:03:40,729 - Core - INFO - ALLoneAPI RPI4 CNN initialization complete.
2020-09-05 07:03:40,730 - Model - INFO - Sending request for: Model/Data/Test/Im028_1.jpg
2020-09-05 07:03:41,681 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 07:03:48,685 - Model - INFO - Sending request for: Model/Data/Test/Im060_1.jpg
2020-09-05 07:03:50,209 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 07:03:57,217 - Model - INFO - Sending request for: Model/Data/Test/Im057_1.jpg
2020-09-05 07:03:58,734 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 07:04:05,742 - Model - INFO - Sending request for: Model/Data/Test/Im041_0.jpg
2020-09-05 07:04:07,258 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 07:04:14,266 - Model - INFO - Sending request for: Model/Data/Test/Im106_0.jpg
2020-09-05 07:04:15,777 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 07:04:22,785 - Model - INFO - Sending request for: Model/Data/Test/Im101_0.jpg
2020-09-05 07:04:24,306 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 07:04:31,314 - Model - INFO - Sending request for: Model/Data/Test/Im088_0.jpg
2020-09-05 07:04:32,822 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)
2020-09-05 07:04:39,830 - Model - INFO - Sending request for: Model/Data/Test/Im026_1.jpg
2020-09-05 07:04:40,650 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 07:04:47,658 - Model - INFO - Sending request for: Model/Data/Test/Im031_1.jpg
2020-09-05 07:04:48,476 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 07:04:55,483 - Model - INFO - Sending request for: Model/Data/Test/Im024_1.jpg
2020-09-05 07:04:56,288 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 07:05:03,296 - Model - INFO - Sending request for: Model/Data/Test/Im099_0.jpg
2020-09-05 07:05:04,797 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 07:05:11,802 - Model - INFO - Sending request for: Model/Data/Test/Im020_1.jpg
2020-09-05 07:05:12,611 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 07:05:19,619 - Model - INFO - Sending request for: Model/Data/Test/Im047_0.jpg
2020-09-05 07:05:21,134 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 07:05:28,142 - Model - INFO - Sending request for: Model/Data/Test/Im053_1.jpg
2020-09-05 07:05:29,654 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly not detected (False Negative)
2020-09-05 07:05:36,661 - Model - INFO - Sending request for: Model/Data/Test/Im006_1.jpg
2020-09-05 07:05:37,478 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 07:05:44,486 - Model - INFO - Sending request for: Model/Data/Test/Im074_0.jpg
2020-09-05 07:05:45,970 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 07:05:52,977 - Model - INFO - Sending request for: Model/Data/Test/Im069_0.jpg
2020-09-05 07:05:54,492 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 07:06:01,499 - Model - INFO - Sending request for: Model/Data/Test/Im063_1.jpg
2020-09-05 07:06:03,005 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 07:06:10,013 - Model - INFO - Sending request for: Model/Data/Test/Im035_0.jpg
2020-09-05 07:06:11,526 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 07:06:18,534 - Model - INFO - Sending request for: Model/Data/Test/Im095_0.jpg
2020-09-05 07:06:19,955 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)
2020-09-05 07:06:26,963 - Model - INFO - Images Classifier: 20
2020-09-05 07:06:26,963 - Model - INFO - True Positives: 9
2020-09-05 07:06:26,964 - Model - INFO - False Positives: 2
2020-09-05 07:06:26,964 - Model - INFO - True Negatives: 8
2020-09-05 07:06:26,965 - Model - INFO - False Negatives: 1
```
We see that we have maintained the same accuracy when testing the server on the CPU.

&nbsp;

# Local Testing (OpenVINO/NCS2)
Now we will use the test data to see how the classifier reacts to our testing data on a Raspberry Pi 4 using OpenVINO and the Neural Compute Stick 2. The command to start testing is as follows:

```
 python ALLOpenVINO.py Classify
```

## Local Testing Results (OpenVINO/NCS2)

```
2020-09-05 06:59:42,670 - Core - INFO - Class initialization complete.
2020-09-05 06:59:42,672 - OpenVINO - INFO - Class initialization complete.
2020-09-05 06:59:42,693 - OpenVINO - INFO - Class initialization complete.
2020-09-05 06:59:42,693 - Core - INFO - Class initialization complete.
2020-09-05 06:59:42,828 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im028_1.jpg
2020-09-05 06:59:44,838 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 2.144590139389038 seconds.
2020-09-05 06:59:45,149 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im060_1.jpg
2020-09-05 06:59:45,171 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.331897497177124 seconds.
2020-09-05 06:59:45,481 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im057_1.jpg
2020-09-05 06:59:45,503 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.3322410583496094 seconds.
2020-09-05 06:59:45,814 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im041_0.jpg
2020-09-05 06:59:45,836 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.33203911781311035 seconds.
2020-09-05 06:59:46,146 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im106_0.jpg
2020-09-05 06:59:46,168 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.3321671485900879 seconds.
2020-09-05 06:59:46,479 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im101_0.jpg
2020-09-05 06:59:46,501 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.33220958709716797 seconds.
2020-09-05 06:59:46,815 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im088_0.jpg
2020-09-05 06:59:46,837 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.3351552486419678 seconds.
2020-09-05 06:59:46,970 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im026_1.jpg
2020-09-05 06:59:46,991 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.15412306785583496 seconds.
2020-09-05 06:59:47,124 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im031_1.jpg
2020-09-05 06:59:47,145 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.15365839004516602 seconds.
2020-09-05 06:59:47,278 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im024_1.jpg
2020-09-05 06:59:47,299 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.15366148948669434 seconds.
2020-09-05 06:59:47,609 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im099_0.jpg
2020-09-05 06:59:47,632 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.33208370208740234 seconds.
2020-09-05 06:59:47,765 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im020_1.jpg
2020-09-05 06:59:47,786 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.15378713607788086 seconds.
2020-09-05 06:59:48,097 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im047_0.jpg
2020-09-05 06:59:48,119 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.3327138423919678 seconds.
2020-09-05 06:59:48,431 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im053_1.jpg
2020-09-05 06:59:48,453 - OpenVINO - INFO - Acute Lymphoblastic Leukemia incorrectly not detected (False Negative) in 0.33391499519348145 seconds.
2020-09-05 06:59:48,586 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im006_1.jpg
2020-09-05 06:59:48,607 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.15400481224060059 seconds.
2020-09-05 06:59:48,915 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im074_0.jpg
2020-09-05 06:59:48,937 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.3291614055633545 seconds.
2020-09-05 06:59:49,247 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im069_0.jpg
2020-09-05 06:59:49,269 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.33180809020996094 seconds.
2020-09-05 06:59:49,580 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im063_1.jpg
2020-09-05 06:59:49,602 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.332273006439209 seconds.
2020-09-05 06:59:49,913 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im035_0.jpg
2020-09-05 06:59:49,935 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.3327963352203369 seconds.
2020-09-05 06:59:50,221 - OpenVINO - INFO - Loaded test image Model/Data/Test/Im095_0.jpg
2020-09-05 06:59:50,243 - OpenVINO - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive) in 0.30745553970336914 seconds.
2020-09-05 06:59:50,243 - OpenVINO - INFO - Images Classifier: 20
2020-09-05 06:59:50,243 - OpenVINO - INFO - True Positives: 9
2020-09-05 06:59:50,244 - OpenVINO - INFO - False Positives: 1
2020-09-05 06:59:50,244 - OpenVINO - INFO - True Negatives: 9
2020-09-05 06:59:50,244 - OpenVINO - INFO - False Negatives: 1
2020-09-05 06:59:50,244 - OpenVINO - INFO - Total Time Taken: 7.541741609573364
```
We see that using OpenVINO and Neural Compute Stick 2 on the Raspberry Pi 4 has increased the processing time by ~4 seconds, and improved the accuracy (there is one less false positive)

&nbsp;

# Server Testing (OpenVINO/NCS2)
Now we will use the test data to see how the server classifier reacts using the CNN model on the Raspberry Pi with OpenVINO and Neural Compute Stick 2.

You need to open two terminal windows or tabs, in the first, use the following command to start the server:

```
python ALLOpenVINO.py Server
```

In your second terminal, use the following command:

```
python ALLOpenVINO.py Client
```

## Server Testing Results (OpenVINO/NCS2)

```
2020-09-05 06:10:17,604 - Core - INFO - Class initialization complete.
2020-09-05 06:10:17,606 - OpenVINO - INFO - Class initialization complete.
2020-09-05 06:10:17,627 - OpenVINO - INFO - Class initialization complete.
2020-09-05 06:10:17,627 - Core - INFO - Class initialization complete.
2020-09-05 06:10:17,628 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im028_1.jpg
2020-09-05 06:10:18,049 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:10:25,057 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im060_1.jpg
2020-09-05 06:10:26,024 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:10:33,031 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im057_1.jpg
2020-09-05 06:10:33,999 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:10:41,007 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im041_0.jpg
2020-09-05 06:10:41,985 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:10:48,993 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im106_0.jpg
2020-09-05 06:10:49,970 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:10:56,978 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im101_0.jpg
2020-09-05 06:10:57,951 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:11:04,958 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im088_0.jpg
2020-09-05 06:11:05,937 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:11:12,944 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im026_1.jpg
2020-09-05 06:11:13,420 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:11:20,427 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im031_1.jpg
2020-09-05 06:11:20,909 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:11:27,916 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im024_1.jpg
2020-09-05 06:11:28,389 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:11:35,396 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im099_0.jpg
2020-09-05 06:11:36,359 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:11:43,367 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im020_1.jpg
2020-09-05 06:11:43,849 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:11:50,856 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im047_0.jpg
2020-09-05 06:11:51,812 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:11:58,819 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im053_1.jpg
2020-09-05 06:11:59,798 - OpenVINO - INFO - Acute Lymphoblastic Leukemia incorrectly not detected (False Negative)
2020-09-05 06:12:06,806 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im006_1.jpg
2020-09-05 06:12:07,294 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:12:14,302 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im074_0.jpg
2020-09-05 06:12:15,269 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:12:22,277 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im069_0.jpg
2020-09-05 06:12:23,253 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:12:30,261 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im063_1.jpg
2020-09-05 06:12:31,229 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:12:38,236 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im035_0.jpg
2020-09-05 06:12:39,204 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:12:46,212 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im095_0.jpg
2020-09-05 06:12:47,118 - OpenVINO - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)
2020-09-05 06:12:54,126 - OpenVINO - INFO - Images Classifier: 20
2020-09-05 06:12:54,126 - OpenVINO - INFO - True Positives: 9
2020-09-05 06:12:54,127 - OpenVINO - INFO - False Positives: 1
2020-09-05 06:12:54,127 - OpenVINO - INFO - True Negatives: 9
2020-09-05 06:12:54,128 - OpenVINO - INFO - False Negatives: 1
```
We see that we have maintained the same accuracy when testing the server on the OpenVINO/NC2.

&nbsp;

# Contributing

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and welcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors

- [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") - [Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss") President/Lead Developer, Sabadell, Spain

&nbsp;

# Versioning

We use SemVer for versioning. For the versions available, see [Releases](../releases "Releases").

&nbsp;

# License

This project is licensed under the **MIT License** - see the [LICENSE](../LICENSE.md "LICENSE") file for details.

&nbsp;

# Bugs/Issues

We use the [repo issues](../issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.