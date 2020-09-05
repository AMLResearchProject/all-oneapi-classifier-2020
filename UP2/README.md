# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
## OneAPI Acute Lymphoblastic Leukemia Classifier
### OneAPI OpenVINO UP2 Acute Lymphoblastic Leukemia Classifier

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
- [Local Testing (OpenVINO)](#local-testing-openvino)
    - [Local Testing Results](#local-testing-results-openvino)
- [Server Testing (OpenVINO/NCS2)](#server-testing-openvinoncs2)
    - [Server Testing Results (OpenVINO/NCS2)](#server-testing-results-openvinoncs2)
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

You will deploy the model to an UP2 and test the classifier with OpenVINO and OpenVINO using the [Intel® Movidius™ Neural Compute Stick 2](https://software.intel.com/content/www/us/en/develop/hardware/neural-compute-stick.html), benchmarking the results.

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

[config.json](Model/config.json "config.json")  holds the configuration for our network. You need to update the **cnn->system->server** field with the IP of your UP2, you should also change the port.

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
            "frozen": "frozen.pb",
            "ir": "Model/IR/frozen.xml"
        }
    }
}
```

The cnn object contains 3 Json Objects (api, data and model) and a JSON Array (core). **api** has the information used to set up your server you will need to add your local ip, **data** has the configuration related to preparing the training and validation data, and **model** holds the model file paths.

&nbsp;

# Local Testing (OpenVINO)
Now we will use the test data to see how the classifier reacts to our testing data on an UP2. Real world testing is the most important testing, as it allows you to see the how the model performs in a real world environment.

This part of the tutorial tests the CNN model on the UP2, OpenVINO and the test data from the **Model/Data/Test** directory. The command to start testing is as follows:

```
 python ALLOpenVINO.py Classify
```

## Local Testing Results (OpenVINO)

```
2020-09-05 06:27:05,360 - Core - INFO - Class initialization complete.
2020-09-05 06:27:05,441 - OpenVINO - INFO - Class initialization complete.
2020-09-05 06:27:05,441 - Core - INFO - Class initialization complete.
2020-09-05 06:27:05,680 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.2389681339263916 seconds.
2020-09-05 06:27:05,909 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.22849607467651367 seconds.
2020-09-05 06:27:06,125 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.21480131149291992 seconds.
2020-09-05 06:27:06,339 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.21379995346069336 seconds.
2020-09-05 06:27:06,449 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.10966968536376953 seconds.
2020-09-05 06:27:06,663 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.21398615837097168 seconds.
2020-09-05 06:27:06,774 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.10968494415283203 seconds.
2020-09-05 06:27:06,988 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.2140810489654541 seconds.
2020-09-05 06:27:07,099 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.10981893539428711 seconds.
2020-09-05 06:27:07,314 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.21475625038146973 seconds.
2020-09-05 06:27:07,597 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.21525812149047852 seconds.
2020-09-05 06:27:07,796 - OpenVINO - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive) in 0.19858908653259277 seconds.
2020-09-05 06:27:07,907 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.1098487377166748 seconds.
2020-09-05 06:27:08,121 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.2133934497833252 seconds.
2020-09-05 06:27:08,337 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.21561598777770996 seconds.
2020-09-05 06:27:08,554 - OpenVINO - INFO - Acute Lymphoblastic Leukemia incorrectly not detected (False Negative) in 0.21682024002075195 seconds.
2020-09-05 06:27:08,767 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.21177411079406738 seconds.
2020-09-05 06:27:08,877 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.10936880111694336 seconds.
2020-09-05 06:27:08,987 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.10971522331237793 seconds.
2020-09-05 06:27:09,202 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.21405744552612305 seconds.
2020-09-05 06:27:09,202 - OpenVINO - INFO - Images Classifier: 20
2020-09-05 06:27:09,203 - OpenVINO - INFO - True Positives: 9
2020-09-05 06:27:09,203 - OpenVINO - INFO - False Positives: 1
2020-09-05 06:27:09,203 - OpenVINO - INFO - True Negatives: 9
2020-09-05 06:27:09,203 - OpenVINO - INFO - False Negatives: 1
2020-09-05 06:27:09,204 - OpenVINO - INFO - Total Time Taken: 3.6825037002563477
```

We see that the testing is than when you tested your model on the Windows machine, but our accuracy accuracy has improved with one less false positive and one more true negative.

&nbsp;

# Server Testing (OpenVINO)
Now we will use the test data to see how the server classifier reacts using the CNN model on the UP2 with OpenVINO.

You need to open two terminal windows or tabs, in the first, use the following command to start the server:

```
python ALLOpenVINO.py Server
```

In your second terminal, use the following command:

```
python ALLOpenVINO.py Client
```

## Server Testing Results (OpenVINO)

```
2020-09-05 06:27:40,836 - Core - INFO - Class initialization complete.
2020-09-05 06:27:40,919 - OpenVINO - INFO - Class initialization complete.
2020-09-05 06:27:40,920 - Core - INFO - Class initialization complete.
2020-09-05 06:27:40,920 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im069_0.jpg
2020-09-05 06:27:41,769 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:27:48,777 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im047_0.jpg
2020-09-05 06:27:49,604 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:27:56,612 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im099_0.jpg
2020-09-05 06:27:57,428 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:28:04,435 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im106_0.jpg
2020-09-05 06:28:05,250 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:28:12,258 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im020_1.jpg
2020-09-05 06:28:12,660 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:28:19,667 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im041_0.jpg
2020-09-05 06:28:20,481 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:28:27,488 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im026_1.jpg
2020-09-05 06:28:27,890 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:28:34,897 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im035_0.jpg
2020-09-05 06:28:35,713 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:28:42,721 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im006_1.jpg
2020-09-05 06:28:43,128 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:28:50,136 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im101_0.jpg
2020-09-05 06:28:50,957 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:28:57,965 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im060_1.jpg
2020-09-05 06:28:58,781 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:29:05,789 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im095_0.jpg
2020-09-05 06:29:06,557 - OpenVINO - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)
2020-09-05 06:29:13,565 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im028_1.jpg
2020-09-05 06:29:13,968 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:29:20,976 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im063_1.jpg
2020-09-05 06:29:21,790 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:29:28,797 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im088_0.jpg
2020-09-05 06:29:29,615 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:29:36,623 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im053_1.jpg
2020-09-05 06:29:37,443 - OpenVINO - INFO - Acute Lymphoblastic Leukemia incorrectly not detected (False Negative)
2020-09-05 06:29:44,450 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im074_0.jpg
2020-09-05 06:29:45,264 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:29:52,271 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im024_1.jpg
2020-09-05 06:29:52,673 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:29:59,681 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im031_1.jpg
2020-09-05 06:30:00,079 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:30:07,087 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im057_1.jpg
2020-09-05 06:30:07,904 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:30:14,912 - OpenVINO - INFO - Images Classifier: 20
2020-09-05 06:30:14,913 - OpenVINO - INFO - True Positives: 9
2020-09-05 06:30:14,913 - OpenVINO - INFO - False Positives: 1
2020-09-05 06:30:14,914 - OpenVINO - INFO - True Negatives: 9
2020-09-05 06:30:14,914 - OpenVINO - INFO - False Negatives: 1
```
We see that we have maintained the same accuracy when testing the server on with OpenVINO.

&nbsp;

# Local Testing (OpenVINO/NCS2)
Now we will use the test data to see how the classifier reacts to our testing data on an UP2 using OpenVINO and the Neural Compute Stick 2.

You need to modify the device in the configuration, change **cnn->model->device** to **MYRIAD**:

```
"model": {
    "device": "MYRIAD",
    "frozen": "frozen.pb",
    "ir": "Model/IR/frozen.xml"
}
```

The command to start testing is as follows:

```
 python ALLOpenVINO.py Classify
```

## Local Testing Results (OpenVINO/NCS2)

```
2020-09-05 06:42:49,978 - Core - INFO - Class initialization complete.
2020-09-05 06:42:51,935 - OpenVINO - INFO - Class initialization complete.
2020-09-05 06:42:51,936 - Core - INFO - Class initialization complete.
2020-09-05 06:42:52,169 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.23211097717285156 seconds.
2020-09-05 06:42:52,384 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.21494746208190918 seconds.
2020-09-05 06:42:52,591 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.20641398429870605 seconds.
2020-09-05 06:42:52,798 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.20645976066589355 seconds.
2020-09-05 06:42:52,897 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.09853148460388184 seconds.
2020-09-05 06:42:53,104 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.20586609840393066 seconds.
2020-09-05 06:42:53,203 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.09820723533630371 seconds.
2020-09-05 06:42:53,456 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.20698928833007812 seconds.
2020-09-05 06:42:53,554 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.09829354286193848 seconds.
2020-09-05 06:42:53,761 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.2067584991455078 seconds.
2020-09-05 06:42:53,968 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.20640182495117188 seconds.
2020-09-05 06:42:54,159 - OpenVINO - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive) in 0.1903209686279297 seconds.
2020-09-05 06:42:54,258 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.09851980209350586 seconds.
2020-09-05 06:42:54,464 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.2062218189239502 seconds.
2020-09-05 06:42:54,673 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.20831537246704102 seconds.
2020-09-05 06:42:54,884 - OpenVINO - INFO - Acute Lymphoblastic Leukemia incorrectly not detected (False Negative) in 0.2100229263305664 seconds.
2020-09-05 06:42:55,089 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative) in 0.20493340492248535 seconds.
2020-09-05 06:42:55,187 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.09807491302490234 seconds.
2020-09-05 06:42:55,286 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.09818243980407715 seconds.
2020-09-05 06:42:55,493 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive) in 0.20639443397521973 seconds.
2020-09-05 06:42:55,493 - OpenVINO - INFO - Images Classifier: 20
2020-09-05 06:42:55,493 - OpenVINO - INFO - True Positives: 9
2020-09-05 06:42:55,493 - OpenVINO - INFO - False Positives: 1
2020-09-05 06:42:55,493 - OpenVINO - INFO - True Negatives: 9
2020-09-05 06:42:55,494 - OpenVINO - INFO - False Negatives: 1
2020-09-05 06:42:55,494 - OpenVINO - INFO - Total Time Taken: 3.5019662380218506
```
We see that using OpenVINO and Neural Compute Stick 2 on the UP2 has increased the processing time by ~0.1 seconds.

&nbsp;

# Server Testing (OpenVINO/NCS2)
Now we will use the test data to see how the server classifier reacts using the CNN model on the UP2 with OpenVINO and Neural Compute Stick 2.

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
2020-09-05 06:47:53,500 - Core - INFO - Class initialization complete.
2020-09-05 06:47:53,502 - OpenVINO - INFO - Class initialization complete.
2020-09-05 06:47:53,502 - Core - INFO - Class initialization complete.
2020-09-05 06:47:53,502 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im069_0.jpg
2020-09-05 06:47:54,374 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:48:01,382 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im047_0.jpg
2020-09-05 06:48:02,224 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:48:09,225 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im099_0.jpg
2020-09-05 06:48:10,056 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:48:17,057 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im106_0.jpg
2020-09-05 06:48:17,888 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:48:24,889 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im020_1.jpg
2020-09-05 06:48:25,291 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:48:32,292 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im041_0.jpg
2020-09-05 06:48:33,130 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:48:40,130 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im026_1.jpg
2020-09-05 06:48:40,533 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:48:47,540 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im035_0.jpg
2020-09-05 06:48:48,373 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:48:55,374 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im006_1.jpg
2020-09-05 06:48:55,777 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:49:02,778 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im101_0.jpg
2020-09-05 06:49:03,618 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:49:10,622 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im060_1.jpg
2020-09-05 06:49:11,456 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:49:18,457 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im095_0.jpg
2020-09-05 06:49:19,246 - OpenVINO - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)
2020-09-05 06:49:26,246 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im028_1.jpg
2020-09-05 06:49:26,648 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:49:33,648 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im063_1.jpg
2020-09-05 06:49:34,483 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:49:41,484 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im088_0.jpg
2020-09-05 06:49:42,321 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:49:49,322 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im053_1.jpg
2020-09-05 06:49:50,157 - OpenVINO - INFO - Acute Lymphoblastic Leukemia incorrectly not detected (False Negative)
2020-09-05 06:49:57,158 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im074_0.jpg
2020-09-05 06:49:57,999 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-05 06:50:05,006 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im024_1.jpg
2020-09-05 06:50:05,408 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:50:12,415 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im031_1.jpg
2020-09-05 06:50:12,815 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:50:19,823 - OpenVINO - INFO - Sending request for: Model/Data/Test/Im057_1.jpg
2020-09-05 06:50:20,660 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-05 06:50:27,661 - OpenVINO - INFO - Images Classifier: 20
2020-09-05 06:50:27,661 - OpenVINO - INFO - True Positives: 9
2020-09-05 06:50:27,662 - OpenVINO - INFO - False Positives: 1
2020-09-05 06:50:27,662 - OpenVINO - INFO - True Negatives: 9
2020-09-05 06:50:27,662 - OpenVINO - INFO - False Negatives: 1
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