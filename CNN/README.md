# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
## OneAPI Acute Lymphoblastic Leukemia Classifier
### ALLoneAPI CNN

![OneAPI Acute Lymphoblastic Leukemia Classifier](../Media/Images/Peter-Moss-Acute-Myeloid-Lymphoblastic-Leukemia-Research-Project.png)

&nbsp;

# Table Of Contents

- [Introduction](#introduction)
- [DISCLAIMER](#disclaimer)
- [ALL-IDB](#all-idb)
  - [ALL_IDB1](#all_idb1)
- [Network Architecture](#network-architecture)
- [Installation](#installation)
- [Getting Started](#getting-started)
    - [Data](#data)
    - [Configuration](#configuration)
- [Metrics](#metrics)
- [Training The Model](#training-the-model)
    - [Start The Training](#start-the-training)
        - [Data](#data)
        - [Model](#model)
        - [Training Results](#training-results)
        - [Metrics Overview](#metrics-overview)
        - [ALL-IDB Required Metrics](#all-idb-required-metrics)
- [Local Testing](#local-testing)
    - [Local Testing Results](#local-testing-results)
- [Server Testing](#server-testing)
    - [Server Testing Results](#server-testing-results)
- [OpenVINO Testing](#openvino-testing)
    - [OpenVINO Testing Results](#openvino-testing-results)
- [Raspberry Pi 4](#raspberry-pi-4)
- [UP2](#up2)
- [Contributing](#contributing)
  - [Contributors](#contributors)
- [Versioning](#versioning)
- [License](#license)
- [Bugs/Issues](#bugs-issues)

&nbsp;

# Introduction
This project trains the model that will be used in our Acute the Lymphoblastic Leukemia Detection Systems. The network provided in this project was originally created in [ALL research papers evaluation project](https://github.com/LeukemiaAiResearch/ALL-IDB-Classifiers/tree/master/Projects/Paper-1 "ALL research papers evaluation project"), where you replicated the network proposed in the [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper by Thanh.TTP, Giao N. Pham, Jin-Hyeok Park, Kwang-Seok Moon, Suk-Hwan Lee, and Ki-Ryong Kwon, and the data augmentation proposed in  [Leukemia Blood Cell Image Classification Using Convolutional Neural Network](http://www.ijcte.org/vol10/1198-H0012.pdf "Leukemia Blood Cell Image Classification Using Convolutional Neural Network") by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon.

We have provided all of the pretrained models in the [Model](Model) directory.

&nbsp;

# DISCLAIMER
These projects should be used for research purposes only. The purpose of the projects is to show the potential of Artificial Intelligence for medical support systems such as diagnosis systems.

Although the classifiers are accurate and show good results both on paper and in real world testing, they are not meant to be an alternative to professional medical diagnosis.

Developers that have contributed to this repository have experience in using Artificial Intelligence for detecting certain types of cancer. They are not doctors, medical or cancer experts.

Please use this system responsibly.

&nbsp;

# ALL-IDB
You need to be granted access to use the Acute Lymphoblastic Leukemia Image Database for Image Processing dataset. You can find the application form and information about getting access to the dataset on [this page](https://homes.di.unimi.it/scotti/all/#download) as youll as information on how to contribute back to the project [here](https://homes.di.unimi.it/scotti/all/results.php). If you are not able to obtain a copy of the dataset please feel free to try this tutorial on your own dataset, you would be very happy to find additional AML & ALL datasets.

## ALL_IDB1
In this project, [ALL-IDB1](https://homes.di.unimi.it/scotti/all/#datasets) is used, one of the datsets from the Acute Lymphoblastic Leukemia Image Database for Image Processing dataset. You will use data augmentation to increase the amount of training and testing data you have.

"The ALL_IDB1 version 1.0 can be used both for testing segmentation capability of algorithms, as youll as the classification systems and image preprocessing methods. This dataset is composed of 108 images collected during September, 2005. It contains about 39000 blood elements, where the lymphocytes has been labeled by expert oncologists. The images are taken with different magnifications of the microscope ranging from 300 to 500."

&nbsp;

# Network Architecture
<img src="https://www.leukemiaresearchfoundation.ai/github/media/images/paper_1_architecture.png" alt="Proposed Architecture" />

_Fig 1. Proposed architecture ([Source](https://airccj.org/CSCP/vol7/csit77505.pdf "Source"))_

In [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System"), the authors propose a simple 5 layer Convolutional Neural Network.

In this project you will use an augmented dataset with the network proposed in this paper, built using Intel Optimized Tensorflow.

You will build a Convolutional Neural Network, as shown in Fig 1, consisting of the following 5 layers (missing out the zero padding layers). Note you are usng an conv sizes of (100x100x30) whereas in the paper, the authors use (50x50x30).

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
Once you have your data you need to add it to the project filesystem. You will notice the data folder in the Model directory, **Model/Data**, inside you have **Train** & **Test**.

You will create an augmented dataset based on the [Leukemia Blood Cell Image Classification Using Convolutional Neural Network](http://www.ijcte.org/vol10/1198-H0012.pdf "Leukemia Blood Cell Image Classification Using Convolutional Neural Network") by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon. In this case, you will use more rotated images to increase the dataset further.

First take the ten positive and ten negative samples shown below, and place them in the **Model/Data/Test** directory. This will be used for testing the model, and also in our detection systems if you use this model with them. To ensure you get the same results, please use the same test images, these same test images are used in our detection systems. You can also try with your own images, but you may not get the same results and if you use this model with any of our detection systems you will need to make some changes when setting them up.

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

Next add the remaining 88 images to the **Model/Data/Train** folder. The test images used will not be augmented.

## Configuration
[config.json](Model/config.json "config.json")  holds the configuration for our network.

```
{
    "cnn": {
        "system": {
            "cores": 8,
            "server": "",
            "port": 1234
        },
        "core": [
            "Train",
            "Server",
            "Client",
            "Classify"
        ],
        "data": {
            "dim": 100,
            "file_type": ".jpg",
            "labels": [0, 1],
            "rotations": 10,
            "seed": 2,
            "split": 0.3,
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
            "train_dir": "Model/Data/Train",
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
            "saved_model_dir": "Model",
            "frozen": "frozen.pb",
            "freezing_log_dir": "Model/Freezing",
            "model": "Model/model.json",
            "weights": "Model/weights.h5"
        },
        "train": {
            "batch": 100,
            "decay_adam": 1e-6,
            "epochs": 150,
            "learning_rate_adam": 1e-4,
            "val_steps": 10
        }
    }
}
```

The cnn object contains 4 Json Objects (api, data, model and train) and a JSON Array (core). Api has the information used to set up your server you will need to add your local ip, data has the configuration related to preparing the training and validation data, model holds the model file paths, and train holds the training parameters.

In my case, the configuration above was the best out of my testing, but you may find different configurations work better. Feel free to update these settings to your liking, and please let us know of your experiences.

&nbsp;

# Metrics
We can use metrics to measure the effectiveness of our model. In this network you will use the following metrics:

```
tf.keras.metrics.BinaryAccuracy(name='accuracy'),
tf.keras.metrics.Precision(name='precision'),
tf.keras.metrics.Recall(name='recall'),
tf.keras.metrics.AUC(name='auc')
```

These metrics will be displayed and plotted once our model is trained.  A useful tutorial while working on the metrics was the [Classification on imbalanced data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data) tutorial on Tensorflow's youbsite.

&nbsp;

# Training the model
Now you are ready to train your model.

## Start The Training
Ensuring you have completed all previous steps, you can start training using the following command.

```
python ALLoneAPI.py Train
```

This tells the classifier to start in Train mode which will start the model training process.

### Data
First the data will be prepared.

```
2020-09-03 22:58:39,427 - Data - INFO - Data shape: (1584, 100, 100, 3)
2020-09-03 22:58:39,428 - Data - INFO - Labels shape: (1584, 2)
2020-09-03 22:58:39,429 - Data - INFO - Raw data: 792
2020-09-03 22:58:39,430 - Data - INFO - Raw negative data: 441
2020-09-03 22:58:39,431 - Data - INFO - Raw positive data: 792
2020-09-03 22:58:39,432 - Data - INFO - Augmented data: (1584, 100, 100, 3)
2020-09-03 22:58:39,432 - Data - INFO - Labels: (1584, 2)
2020-09-03 22:58:39,620 - Data - INFO - Training data: (1180, 100, 100, 3)
2020-09-03 22:58:39,620 - Data - INFO - Training labels: (1180, 2)
2020-09-03 22:58:39,622 - Data - INFO - Validation data: (404, 100, 100, 3)
2020-09-03 22:58:39,623 - Data - INFO - Validation labels: (404, 2)
2020-09-03 22:58:39,647 - Model - INFO - Data preperation complete.
```

### Model Summary
Our network matches the architecture proposed in the paper exactly, with exception to maybe the optimizer and loss function as this info was not provided in the paper.

Before the model begins training, you will be shown the model summary, or architecture.

```
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
2020-09-03 07:46:38,579 - Model - INFO - Network initialization complete.
2020-09-03 07:46:38,579 - Model - INFO - Using Adam Optimizer.
Train on 1180 samples, validate on 404 samples
```

## Training Results
Below are the training results for 150 epochs. Here there is an issue with the log files being written as we are using multiple threads, there will be many errors shown in the console but you can ignore them. Hidden in the errors you will find the metrics.

<img src="Model/Plots/Accuracy.png" alt="Adam Optimizer Results" />

_Fig 2. Accuracy_

<img src="Model/Plots/Loss.png" alt="Ubuntu/GTX 1050 ti Loss" />

_Fig 3. Loss_

<img src="Model/Plots/Precision.png" alt="Ubuntu/GTX 1050 ti Precision" />

_Fig 4. Precision_

<img src="Model/Plots/Recall.png" alt="Ubuntu/GTX 1050 ti Recall" />

_Fig 5. Recall_

<img src="Model/Plots/AUC.png" alt="Ubuntu/GTX 1050 ti AUC" />

_Fig 6. AUC_

```
2020-09-04 09:34:51,246 - Model - INFO - Metrics: loss 0.06714094072432801
2020-09-04 09:34:51,266 - Model - INFO - Metrics: acc 0.9752475
2020-09-04 09:34:51,287 - Model - INFO - Metrics: precision 0.9752475
2020-09-04 09:34:51,304 - Model - INFO - Metrics: recall 0.9752475
2020-09-04 09:34:51,326 - Model - INFO - Metrics: auc 0.9972336
2020-09-04 09:36:49,950 - Model - INFO - Confusion Matrix: [[224   4]
2020-09-04 09:36:56,044 - Model - INFO - True Positives: 170(42.07920792079208%)
2020-09-04 09:36:56,121 - Model - INFO - False Positives: 4(0.9900990099009901%)
2020-09-04 09:36:56,183 - Model - INFO - True Negatives: 224(55.445544554455445%)
2020-09-04 09:36:56,266 - Model - INFO - False Negatives: 6(1.4851485148514851%)
2020-09-04 09:36:56,325 - Model - INFO - Specificity: 0.9824561403508771
2020-09-04 09:36:56,391 - Model - INFO - Misclassification: 10(2.4752475247524752%)
```

## Metrics Overview
| Accuracy | Recall | Precision | AUC/ROC |
| ---------- | ---------- | ---------- | ---------- |
| 0.9752475 | 0.9752475 | 0.9752475 | 0.9972336 |

## ALL-IDB Required Metrics
| Figures of merit     | Amount/Value | Percentage |
| -------------------- | ----- | ---------- |
| True Positives       | 170 | 42.07920792079208% |
| False Positives      | 4 | 0.9900990099009901% |
| True Negatives       | 224 | 55.445544554455445% |
| False Negatives      | 6 | 1.4851485148514851% |
| Misclassification    | 10 | 2.4752475247524752% |
| Sensitivity / Recall | 0.9752475   | 0.98% |
| Specificity          | 0.9824561403508771  | 98% |

&nbsp;

# Local Testing
Now you will use the test data to see how the classifier reacts to our testing data. Real world testing is the most important testing, as it allows you to see the how the model performs in a real world environment.

This part of the system will use the test data from the **Model/Data/Test** directory. The command to start testing locally is as follows:

```
python ALLoneAPI.py Classify
```

## Local Testing Results

```
Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) solvers for sklearn enabled: https://intelpython.github.io/daal4py/sklearn.html
2020-09-04 09:43:45,406 - Core - INFO - Helpers class initialization complete.
2020-09-04 09:43:45,410 - Model - INFO - Model class initialization complete.
2020-09-04 09:43:45,410 - Core - INFO - ALLoneAPI CNN initialization complete.
2020-09-04 09:43:45.437314: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-09-04 09:43:45,617 - Model - INFO - Model loaded
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
2020-09-04 09:43:45,699 - Model - INFO - Loaded test image Model/Data/Test/Im006_1.jpg
2020-09-04 09:43:45,781 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 09:43:45,848 - Model - INFO - Loaded test image Model/Data/Test/Im020_1.jpg
2020-09-04 09:43:45,864 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 09:43:45,934 - Model - INFO - Loaded test image Model/Data/Test/Im024_1.jpg
2020-09-04 09:43:45,951 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 09:43:46,022 - Model - INFO - Loaded test image Model/Data/Test/Im026_1.jpg
2020-09-04 09:43:46,038 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 09:43:46,108 - Model - INFO - Loaded test image Model/Data/Test/Im028_1.jpg
2020-09-04 09:43:46,122 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 09:43:46,187 - Model - INFO - Loaded test image Model/Data/Test/Im031_1.jpg
2020-09-04 09:43:46,200 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 09:43:46,363 - Model - INFO - Loaded test image Model/Data/Test/Im035_0.jpg
2020-09-04 09:43:46,380 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 09:43:46,540 - Model - INFO - Loaded test image Model/Data/Test/Im041_0.jpg
2020-09-04 09:43:46,557 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 09:43:46,700 - Model - INFO - Loaded test image Model/Data/Test/Im047_0.jpg
2020-09-04 09:43:46,718 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 09:43:46,881 - Model - INFO - Loaded test image Model/Data/Test/Im053_1.jpg
2020-09-04 09:43:46,896 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly not detected (False Negative)
2020-09-04 09:43:47,046 - Model - INFO - Loaded test image Model/Data/Test/Im057_1.jpg
2020-09-04 09:43:47,062 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 09:43:47,206 - Model - INFO - Loaded test image Model/Data/Test/Im060_1.jpg
2020-09-04 09:43:47,222 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 09:43:47,384 - Model - INFO - Loaded test image Model/Data/Test/Im063_1.jpg
2020-09-04 09:43:47,400 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 09:43:47,542 - Model - INFO - Loaded test image Model/Data/Test/Im069_0.jpg
2020-09-04 09:43:47,557 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 09:43:47,716 - Model - INFO - Loaded test image Model/Data/Test/Im074_0.jpg
2020-09-04 09:43:47,732 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 09:43:47,889 - Model - INFO - Loaded test image Model/Data/Test/Im088_0.jpg
2020-09-04 09:43:47,906 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)
2020-09-04 09:43:48,041 - Model - INFO - Loaded test image Model/Data/Test/Im095_0.jpg
2020-09-04 09:43:48,056 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)
2020-09-04 09:43:48,216 - Model - INFO - Loaded test image Model/Data/Test/Im099_0.jpg
2020-09-04 09:43:48,231 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 09:43:48,385 - Model - INFO - Loaded test image Model/Data/Test/Im101_0.jpg
2020-09-04 09:43:48,401 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 09:43:48,552 - Model - INFO - Loaded test image Model/Data/Test/Im106_0.jpg
2020-09-04 09:43:48,568 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 09:43:48,568 - Model - INFO - Images Classifier: 20
2020-09-04 09:43:48,569 - Model - INFO - True Positives: 9
2020-09-04 09:43:48,570 - Model - INFO - False Positives: 2
2020-09-04 09:43:48,571 - Model - INFO - True Negatives: 8
2020-09-04 09:43:48,573 - Model - INFO - False Negatives: 1
```

&nbsp;

# Server Testing
Now you will use the test data to see how the server classifier reacts.

This part of the system will use the test data from the **Model/Data/Test** directory.

You need to open two terminal windows or tabs, in the first, use the following command to start the server:

```
python ALLoneAPI.py Server
```
You should see the following:
```
Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) solvers for sklearn enabled: https://intelpython.github.io/daal4py/sklearn.html
2020-09-04 09:46:52,669 - Core - INFO - Helpers class initialization complete.
2020-09-04 09:46:52,671 - Model - INFO - Model class initialization complete.
2020-09-04 09:46:52,672 - Core - INFO - ALLoneAPI CNN initialization complete.
2020-09-04 09:46:52.707819: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-09-04 09:46:52,915 - Model - INFO - Model loaded
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
 * Serving Flask app "Classes.Server" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://192.168.1.33:1234/ (Press CTRL+C to quit)
```

In your second terminal, use the following command:

```
python ALLoneAPI.py Client
```

## Server Testing Results

```
python ALLoneAPI.py Client
Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) solvers for sklearn enabled: https://intelpython.github.io/daal4py/sklearn.html
2020-09-04 10:16:49,810 - Core - INFO - Helpers class initialization complete.
2020-09-04 10:16:49,812 - Model - INFO - Model class initialization complete.
2020-09-04 10:16:49,812 - Core - INFO - ALLoneAPI CNN initialization complete.
2020-09-04 10:16:49,814 - Model - INFO - Sending request for: Model/Data/Test/Im006_1.jpg
2020-09-04 10:17:01,939 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-09-04 10:17:08,942 - Model - INFO - Sending request for: Model/Data/Test/Im020_1.jpg
2020-09-04 10:17:09,209 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-09-04 10:17:16,210 - Model - INFO - Sending request for: Model/Data/Test/Im024_1.jpg
2020-09-04 10:17:16,452 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-09-04 10:17:23,454 - Model - INFO - Sending request for: Model/Data/Test/Im026_1.jpg
2020-09-04 10:17:23,698 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-09-04 10:17:30,700 - Model - INFO - Sending request for: Model/Data/Test/Im028_1.jpg
2020-09-04 10:17:30,952 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-09-04 10:17:37,953 - Model - INFO - Sending request for: Model/Data/Test/Im031_1.jpg
2020-09-04 10:17:38,213 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-09-04 10:17:45,215 - Model - INFO - Sending request for: Model/Data/Test/Im035_0.jpg
2020-09-04 10:17:45,787 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-09-04 10:17:52,789 - Model - INFO - Sending request for: Model/Data/Test/Im041_0.jpg
2020-09-04 10:17:53,349 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-09-04 10:18:00,350 - Model - INFO - Sending request for: Model/Data/Test/Im047_0.jpg
2020-09-04 10:18:00,891 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-09-04 10:18:07,893 - Model - INFO - Sending request for: Model/Data/Test/Im053_1.jpg
2020-09-04 10:18:08,780 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly not detected (False Negative)

2020-09-04 10:18:15,782 - Model - INFO - Sending request for: Model/Data/Test/Im057_1.jpg
2020-09-04 10:18:16,354 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-09-04 10:18:23,356 - Model - INFO - Sending request for: Model/Data/Test/Im060_1.jpg
2020-09-04 10:18:23,916 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-09-04 10:18:30,919 - Model - INFO - Sending request for: Model/Data/Test/Im063_1.jpg
2020-09-04 10:18:31,462 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-09-04 10:18:38,464 - Model - INFO - Sending request for: Model/Data/Test/Im069_0.jpg
2020-09-04 10:18:38,998 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-09-04 10:18:46,001 - Model - INFO - Sending request for: Model/Data/Test/Im074_0.jpg
2020-09-04 10:18:46,555 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-09-04 10:18:53,557 - Model - INFO - Sending request for: Model/Data/Test/Im088_0.jpg
2020-09-04 10:18:54,187 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)

2020-09-04 10:19:01,189 - Model - INFO - Sending request for: Model/Data/Test/Im095_0.jpg
2020-09-04 10:19:01,817 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)

2020-09-04 10:19:08,821 - Model - INFO - Sending request for: Model/Data/Test/Im099_0.jpg
2020-09-04 10:19:09,411 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-09-04 10:19:16,414 - Model - INFO - Sending request for: Model/Data/Test/Im101_0.jpg
2020-09-04 10:19:16,978 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-09-04 10:19:23,980 - Model - INFO - Sending request for: Model/Data/Test/Im106_0.jpg
2020-09-04 10:19:24,527 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-09-04 10:19:31,529 - Model - INFO - Images Classifier: 20
2020-09-04 10:19:31,529 - Model - INFO - True Positives: 9
2020-09-04 10:19:31,530 - Model - INFO - False Positives: 2
2020-09-04 10:19:31,531 - Model - INFO - True Negatives: 8
2020-09-04 10:19:31,531 - Model - INFO - False Negatives: 1
```

&nbsp;

# OpenVINO Testing
At the end of the training the program will freeze the Tensorflow model ready for converting to an Intermediate Representation so that the model can be used with OpenVINO.

In the [Model](Model) directory, you will find the model files and a directory called **Freezing**. Inside this directory is the frozen model that we will convert. To convert the model to IR, use the following commands, replacing **PathToProjectRoot** with the path to the CNN folder on your computer:

```
  cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
  setupvars.bat
  cd C:\Program Files (x86)\IntelSWTools\openvino\bin\deployment_tools\model_optimizer
  python3 mo_tf.py --input_model PathToProjectRoot\Model\Freezing\frozen.pb --input_shape [1,100,100,3] --output_dir PathToProjectRoot\Model\IR --reverse_input_channels
```

This will create the Intermediate Representation and you can now move on to testing the IR with OpenVINO. This part of the system will use the test data from the **Model/Data/Test** directory. The command to start testing is as follows:

```
python OpenVINO.py
```

## OpenVINO Testing Results

```
2020-09-04 11:36:59,370 - OpenVINO - INFO - Helpers class initialization complete.
2020-09-04 11:36:59,371 - OpenVINO - INFO - ALLoneAPI OpenVINO initialization complete.
2020-09-04 11:36:59,695 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 11:36:59,916 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 11:37:00,130 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 11:37:00,344 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 11:37:00,454 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 11:37:00,669 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 11:37:00,779 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 11:37:01,142 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 11:37:01,252 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 11:37:01,466 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 11:37:01,680 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 11:37:01,879 - OpenVINO - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)
2020-09-04 11:37:01,989 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 11:37:02,203 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 11:37:02,419 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 11:37:02,634 - OpenVINO - INFO - Acute Lymphoblastic Leukemia incorrectly not detected (False Negative)
2020-09-04 11:37:02,847 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-09-04 11:37:02,957 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 11:37:03,067 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 11:37:03,282 - OpenVINO - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-09-04 11:37:03,282 - OpenVINO - INFO - Images Classifier: 20
2020-09-04 11:37:03,282 - OpenVINO - INFO - True Positives: 9
2020-09-04 11:37:03,282 - OpenVINO - INFO - False Positives: 1
2020-09-04 11:37:03,282 - OpenVINO - INFO - True Negatives: 9
2020-09-04 11:37:03,282 - OpenVINO - INFO - False Negatives: 1
```

&nbsp;

# Raspberry Pi 4
Now that your model is trained and tested, head over to the [RPI 4](../RPI4 "RPI 4") project to setup your model on the Raspberry Pi 4 ready to be used with the [HIAS ALL Detection System](https://github.com/LeukemiaAiResearch/HIAS "HIAS ALL Detection System") or our other detection systems.

&nbsp;

# UP2
If you want to use an UP2 head to the [UP2](../UP2 "UP2") project to setup your model on the UP2 ready to be used with the [HIAS ALL Detection System](https://github.com/LeukemiaAiResearch/HIAS "HIAS ALL Detection System") or our other detection systems.

&nbsp;

# Contributing
The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and youlcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors
- [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") - [Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss") President/Lead Developer, Sabadell, Spain

&nbsp;

# Versioning
You use SemVer for versioning. For the versions available, see [Releases](../releases "Releases").

&nbsp;

# License
This project is licensed under the **MIT License** - see the [LICENSE](../LICENSE.md "LICENSE") file for details.

&nbsp;

# Bugs/Issues
You use the [repo issues](../issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.