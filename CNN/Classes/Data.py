############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    OneAPI Acute Lymphoblastic Leukemia Classifier
# Project:       OneAPI Acute Lymphoblastic Leukemia Classifier CNN
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
#
# Title:         Data helper class
# Description:   Data functions for the OneAPI Acute OneAPI Acute Lymphoblastic Leukemia Classifier CNN.
# License:       MIT License
# Last Modified: 2020-09-04
#
############################################################################################

import cv2
import pathlib
import random
import os

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from scipy import ndimage
from skimage import transform as tm

from Classes.Helpers import Helpers
from Classes.Augmentation import Augmentation


class Data():
	""" Data class

	Data functions for the OneAPI Acute Lymphoblastic Leukemia Classifier CNN.
	"""

	def __init__(self):
		""" Initializes the class. """

		self.Helpers = Helpers("Data", False)

		self.seed = self.Helpers.confs["cnn"]["data"]["seed"]
		self.dim = self.Helpers.confs["cnn"]["data"]["dim"]

		seed(self.seed)
		random.seed(self.seed)

		self.data = []
		self.labels = []

		self.Helpers.logger.info("Class initialization complete.")

	def do_im_process(self):
		""" Sorts the training data and labels for your model. """

		aug = Augmentation()

		data_dir = pathlib.Path(
			self.Helpers.confs["cnn"]["data"]["train_dir"])
		data = list(data_dir.glob(
			'*' + self.Helpers.confs["cnn"]["data"]["file_type"]))

		count = 0
		neg_count = 0
		pos_count = 0

		augmented_data = []
		augmented_labels = []

		for rimage in data:
			fpath = str(rimage)
			fname = os.path.basename(rimage)
			label = 0 if "_0" in fname else 1

			image = self.resize(fpath, self.dim)

			if image.shape[2] == 1:
				image = np.dstack(
					[image, image, image])

			augmented_data.append(image.astype(np.float32)/255.)
			augmented_labels.append(label)

			augmented_data.append(aug.grayscale(image))
			augmented_labels.append(label)

			augmented_data.append(aug.equalize_hist(image))
			augmented_labels.append(label)

			horizontal, vertical = aug.reflection(image)
			augmented_data.append(horizontal)
			augmented_labels.append(label)

			augmented_data.append(vertical)
			augmented_labels.append(label)

			augmented_data.append(aug.gaussian(image))
			augmented_labels.append(label)

			augmented_data.append(aug.translate(image))
			augmented_labels.append(label)

			augmented_data.append(aug.shear(image))
			augmented_labels.append(label)

			self.data, self.labels = aug.rotation(image, label, augmented_data, augmented_labels)

			if "_0" in fname:
				neg_count += 9
			else:
				pos_count += 9
			count += 9

		self.shuffle()
		self.convert_data()
		self.encode_labels()

		self.Helpers.logger.info("Raw data: " + str(count))
		self.Helpers.logger.info("Raw negative data: " + str(neg_count))
		self.Helpers.logger.info("Raw positive data: " + str(count))
		self.Helpers.logger.info("Augmented data: " + str(self.data.shape))
		self.Helpers.logger.info("Labels: " + str(self.labels.shape))

		self.get_split()

	def convert_data(self):
		""" Converts the training data to a numpy array. """

		self.data = np.array(self.data)
		self.Helpers.logger.info("Data shape: " + str(self.data.shape))

	def encode_labels(self):
		""" One Hot Encodes the labels. """

		encoder = OneHotEncoder(categories='auto')

		self.labels = np.reshape(self.labels, (-1, 1))
		self.labels = encoder.fit_transform(self.labels).toarray()
		self.Helpers.logger.info("Labels shape: " + str(self.labels.shape))

	def shuffle(self):
		""" Shuffles the data and labels. """

		self.data, self.labels = shuffle(self.data, self.labels, random_state=self.seed)

	def get_split(self):
		""" Splits the data and labels creating training and validation datasets. """

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			self.data, self.labels, test_size=0.255, random_state=self.seed)

		self.Helpers.logger.info("Training data: " + str(self.X_train.shape))
		self.Helpers.logger.info("Training labels: " + str(self.y_train.shape))
		self.Helpers.logger.info("Validation data: " + str(self.X_test.shape))
		self.Helpers.logger.info("Validation labels: " + str(self.y_test.shape))

	def resize(self, path, dim):
		""" Resizes an image to the provided dimensions (dim). """

		return cv2.resize(cv2.imread(path), (dim, dim))
