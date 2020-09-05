# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
## OneAPI Acute Lymphoblastic Leukemia Classifier
### OneAPI OpenVINO UP2 Acute Lymphoblastic Leukemia Classifier

![OneAPI OpenVINO UP2 Acute Lymphoblastic Leukemia Classifier](../../Media/Images/Peter-Moss-Acute-Myeloid-Lymphoblastic-Leukemia-Research-Project.png)

&nbsp;

# Table Of Contents

- [Installation](#installation)
	- [Prerequisites](#prerequisites)
      - [OneAPI Acute Lymphoblastic Leukemia Classifier CNN](#oneapi-acute-lymphoblastic-leukemia-classifier-cnn)
      - [Intermediate Representation](#intermediate-representation)
      - [Ubuntu 18.04](#ubuntu-1804)
    - [Intel® Distribution of OpenVINO™ Toolkit](#intel-distribution-of-openvino-toolkit)
      - [Intel® Movidius™ Neural Compute Stick 2](#intel-movidius-neural-compute-stick-2)
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

## Prerequisites
Before you can install this project, there are some prerequisites.

### OneAPI Acute Lymphoblastic Leukemia Classifier CNN
For this project you will use the model created in the [CNN](../../CNN "CNN") project. If you would like to train your own model you can follow the CNN guide, or you can use the pre-trained model and weights provided in the [Model](../Model "Model") directory.

### Intermediate Representation
If you are training the model yourself, you need to convert your model to an Intermediate Representation so that it can be used with OpenVINO and the Neural Compute Stick 2.

To do this, once you have finished the OneAPI Acute Lymphoblastic Leukemia Classifier CNN tutorial, use the following commands, replacing **PathToProject** with the path to the CNN project:

```
 cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
  setupvars.bat
  cd C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer
 python3 mo_tf.py --input_model PathToProject\Model\Freezing\frozen.pb --input_shape [1,100,100,3] --output_dir PathToProject\Model\IR --reverse_input_channels
```
The Intermediate Representation for your model will now be accessible in the [CNN project IR directory](../../CNN/Model/IR), you need to copy these files to your UP2 in the same location in the UP2 Model directory.

### Ubuntu 18.04
For this Project, the operating system choice is [Ubuntu 18.04](https://releases.ubuntu.com/18.04/ "Ubuntu 18.04").

## Intel® Distribution of OpenVINO™ Toolkit
To install Intel® Distribution of OpenVINO™ Toolkit, follow the steps on [this link](https://software.seek.intel.com/openvino-toolkit?os=linux) to download OpenVINO, making sure you choose 2020.4.

Make sure the compressed folder is in you user home directory and use the following steps:

```
  tar -xvzf l_openvino_toolkit_p_2020.4.287.tgz
  cd l_openvino_toolkit_p_2020.4.287
  sudo ./install.sh
```

Follow the installation guide, once you have accepted the End User License and concented, or not consented to the collection of your data, the script will check the prerequisites.

When you are told about missing dependencies. choose **1** to **Skip prerequisites** and then **1** again, and once more to **Skip prerequisites**.

When instructed to, press **Enter** to quit.

Now we need to update our **.bashrc** file so that OpenVINO loads every time you open a terminal.

In your user home directory, use the following command:
```
  nano ~/.bashrc
```
This will open up the file in Nano. Scroll to the bottom and add:

```
  # OpenVINO
  source /opt/intel/openvino/bin/setupvars.sh
```
Save and close the file then use the following command to source the .bashrc file:
```
  source ~/.bashrc
```
You will see the following:
```
  [setupvars.sh] OpenVINO environment initialized
```
And now we will configure the model optimizer:
```
  cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites
  sudo ./install_prerequisites.sh
```
### Intel® Movidius™ Neural Compute Stick and Intel® Neural Compute Stick 2
Now we will set up ready for Neural Compute Stick and Neural Compute Stick 2.
```
  sudo usermod -a -G users "$(whoami)"
```
Now close your existing terminal and open a new open. Once in your new terminal use the following commands:
```
  sudo cp /opt/intel/openvino/inference_engine/external/97-myriad-usbboot.rules /etc/udev/rules.d/
  sudo udevadm control --reload-rules
  sudo udevadm trigger
  sudo ldconfig
```

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

Navigate to **oneAPI-ALL-Classifier/UP2** directory, this is your project root directory for this tutorial.

### Developer Forks

Developers from the Github community that would like to contribute to the development of this project should first create a fork, and clone that repository. For detailed information please view the [CONTRIBUTING](../../CONTRIBUTING.md "CONTRIBUTING") guide. You should pull the latest code from the development branch.

```
 git clone -b "0.4.0" https://github.com/AMLResearchProject/oneAPI-ALL-Classifier.git
```

The **-b "0.4.0"** parameter ensures you get the code from the latest master branch. Before using the below command please check our latest master branch in the button at the top of the project README.

## Setup File

All other requirements are included in **Setup.sh**. You can run this file on machine by navigating to the **UP2** directory in terminal and using the commands below:

```
 sed -i 's/\r//' Setup.sh
 sh Setup.sh
```

# Continue
Now you can continue with the [OneAPI OpenVINO UP2 Acute Lymphoblastic Leukemia Classifier tutorial](../README.md#getting-started)

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