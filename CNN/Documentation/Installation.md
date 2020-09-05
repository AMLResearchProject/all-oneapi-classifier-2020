# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
## OneAPI Acute Lymphoblastic Leukemia Classifier
### ALLoneAPI CNN Installation

![OneAPI Acute Lymphoblastic Leukemia Classifier](../../Media/Images/Peter-Moss-Acute-Myeloid-Lymphoblastic-Leukemia-Research-Project.png)

&nbsp;

# Table Of Contents

- [Installation](#installation)
	- [Anaconda](#anaconda)
	- [Intel® Optimization for TensorFlow](#intel-optimization-for-tensorflow)
	- [Intel® Distribution of OpenVINO™ Toolkit](#intel-distribution-of-openvino-toolkit)
	- [Clone The Repository](#clone-the-repository)
	- [Setup File](#setup-file)
	- [Continue](#continue)
- [Contributing](#contributing)
  - [Contributors](#contributors)
- [Versioning](#versioning)
- [License](#license)
- [Bugs/Issues](#bugs-issues)

# Installation
This guide will take you through installing the requirements for the OneAPI Acute Lymphoblastic Leukemia Classifier.

## Anaconda
If you haven't already installed Anaconda you will need to install it now. Follow the [Anaconda installation guide](https://docs.anaconda.com/anaconda/install/ "Anaconda installation guide") to do so.

## Tensorflow GPU
Now you will install Tensorflow GPU in an environment.

```
conda create --name tf2gpu tensorflow-gpu
```

## Intel® Optimization for TensorFlow
Now you will install the Intel® Optimization for TensorFlow using Anaconda.

```
conda create -n tfmkl python=3
conda activate tfmkl
conda install tensorflow-mkl
conda deactivate
```

## Intel® Distribution of OpenVINO™ Toolkit
To install Intel® Distribution of OpenVINO™ Toolkit, follow [this tutorial](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html)

## Clone the repository

Clone the [OneAPI Acute Lymphoblastic Leukemia Classifier](https://github.com/AMLResearchProject/oneAPI-ALL-Classifier " OneAPI Acute Lymphoblastic Leukemia Classifier") repository from the [Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project](https://github.com/AMLResearchProject "Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project") Github Organization.

To clone the repository and install the OneAPI Acute Lymphoblastic Leukemia Classifier Classifier, make sure you have Git installed. Now navigate to the a directory on your device using commandline, and then use the following command.

```
 git clone https://github.com/AMLResearchProject/oneAPI-ALL-Classifier.git
```

Once you have used the command above you will see a directory called **oneAPI-ALL-Classifier** in your home directory.

```
 ls
```

Using the ls command in your home directory should show you the following.

```
 oneAPI-ALL-Classifier
```

Navigate to **oneAPI-ALL-Classifier/CNN** directory, this is your project root directory for this tutorial.

### Developer Forks

Developers from the Github community that would like to contribute to the development of this project should first create a fork, and clone that repository. For detailed information please view the [CONTRIBUTING](../../CONTRIBUTING.md "CONTRIBUTING") guide. You should pull the latest code from the development branch.

```
 git clone -b "0.2.0" https://github.com/AMLResearchProject/oneAPI-ALL-Classifier.git
```

The **-b "0.2.0"** parameter ensures you get the code from the latest master branch. Before using the below command please check our latest master branch in the button at the top of the project README.

## Setup File

All other requirements are included in **Setup.sh**. You can run this file on your machine by navigating to the **CNN** directory in terminal and using the commands below:

```
 conda activate tf2gpu
 sh Setup.sh
 conda deactivate
```
```
 conda activate tfmkl
 sh Setup.sh
 conda deactivate
```

# Continue
First activate your Anaconda environment:
```
 conda activate tfmkl
```
Now you can continue with the [oneAPI ALL Classifier tutorial](../README.md#getting-started)

&nbsp;

# Contributing

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and youlcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](../../CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors

- [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") - [Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss") President/Lead Developer, Sabadell, Spain

&nbsp;

# Versioning

You use SemVer for versioning. For the versions available, see [Releases](../../releases "Releases").

&nbsp;

# License

This project is licensed under the **MIT License** - see the [LICENSE](../../LICENSE.md "LICENSE") file for details.

&nbsp;

# Bugs/Issues

You use the [repo issues](../../issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](../../CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.