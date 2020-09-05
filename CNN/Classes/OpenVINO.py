############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    OneAPI Acute Lymphoblastic Leukemia Classifier
# Project:       OneAPI Acute Lymphoblastic Leukemia Classifier CNN
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
#
# Title:         Model Class
# Description:   Model functions for the OneAPI Acute Lymphoblastic Leukemia Classifier CNN.
# License:       MIT License
# Last Modified: 2020-09-04
#
############################################################################################

import cv2
import json
import os
import requests
import time

import numpy as np

from openvino import inference_engine as ie
from openvino.inference_engine import IENetwork, IECore

from io import BytesIO
from PIL import Image

from Classes.Helpers import Helpers

class OpenVINO():
	""" Model Class

	Model functions for the OneAPI Acute Lymphoblastic Leukemia Classifier CNN.
	"""

	def __init__(self):
		""" Initializes the class. """

		self.Helpers = Helpers("OpenVINO", False)

		os.environ["KMP_BLOCKTIME"] = "1"
		os.environ["KMP_SETTINGS"] = "1"
		os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
		os.environ["OMP_NUM_THREADS"] = str(self.Helpers.confs["cnn"]["system"]["cores"])

		self.testing_dir = self.Helpers.confs["cnn"]["data"]["test"]
		self.valid = self.Helpers.confs["cnn"]["data"]["valid_types"]

		mxml = self.Helpers.confs["cnn"]["model"]["ir"]
		mbin = os.path.splitext(mxml)[0] + ".bin"

		ie = IECore()
		self.net = ie.read_network(model = mxml, weights = mbin)
		self.input_blob = next(iter(self.net.inputs))
		self.net = ie.load_network(network=self.net,
						device_name=self.Helpers.confs["cnn"]["model"]["device"])

		self.Helpers.logger.info("Class initialization complete.")

	def test_classifier(self):
		""" Tests the trained model. """

		files = 0
		tp = 0
		fp = 0
		tn = 0
		fn = 0
		totaltime = 0

		for testFile in os.listdir(self.testing_dir):
			if os.path.splitext(testFile)[1] in self.valid:

				files += 1
				fileName = self.testing_dir + "/" + testFile

				start = time.time()
				img = Image.open(fileName)
				processed = self.reshape(img)
				prediction = self.get_predictions(processed)
				end = time.time()
				benchmark = end - start
				totaltime += benchmark

				msg = ""
				if prediction == 1 and "_1." in testFile:
					tp += 1
					msg = "Acute Lymphoblastic Leukemia correctly detected (True Positive) in " + str(benchmark) + " seconds."
				elif prediction == 1 and "_0." in testFile:
					fp += 1
					msg = "Acute Lymphoblastic Leukemia incorrectly detected (False Positive) in " + str(benchmark) + " seconds."
				elif prediction == 0 and "_0." in testFile:
					tn += 1
					msg = "Acute Lymphoblastic Leukemia correctly not detected (True Negative) in " + str(benchmark) + " seconds."
				elif prediction == 0 and "_1." in testFile:
					fn += 1
					msg = "Acute Lymphoblastic Leukemia incorrectly not detected (False Negative) in " + str(benchmark) + " seconds."
				self.Helpers.logger.info(msg)

		self.Helpers.logger.info("Images Classifier: " + str(files))
		self.Helpers.logger.info("True Positives: " + str(tp))
		self.Helpers.logger.info("False Positives: " + str(fp))
		self.Helpers.logger.info("True Negatives: " + str(tn))
		self.Helpers.logger.info("False Negatives: " + str(fn))
		self.Helpers.logger.info("Total Time Taken: " + str(totaltime))

	def send_request(self, img_path):
		""" Sends image to the inference API endpoint. """

		self.Helpers.logger.info("Sending request for: " + img_path)

		_, img_encoded = cv2.imencode('.jpg', cv2.imread(img_path))
		response = requests.post(
			self.addr, data=img_encoded.tostring(), headers=self.headers)
		response = json.loads(response.text)

		return response

	def test_http_classifier(self):
		""" Tests the trained model via HTTP. """

		msg = ""

		files = 0
		tp = 0
		fp = 0
		tn = 0
		fn = 0

		self.addr = "http://" + self.Helpers.confs["cnn"]["system"]["server"] + \
			':'+str(self.Helpers.confs["cnn"]["system"]["port"]) + '/Inference'
		self.headers = {'content-type': 'image/jpeg'}

		for data in os.listdir(self.testing_dir):
			if os.path.splitext(data)[1] in self.valid:

				response = self.send_request(self.testing_dir + "/" + data)

				msg = ""
				if response["Diagnosis"] == "Positive" and "_1." in data:
					tp += 1
					msg = "Acute Lymphoblastic Leukemia correctly detected (True Positive)"
				elif response["Diagnosis"] == "Positive" and "_0." in data:
					fp += 1
					msg = "Acute Lymphoblastic Leukemia incorrectly detected (False Positive)"
				elif response["Diagnosis"] == "Negative" and "_0." in data:
					tn += 1
					msg = "Acute Lymphoblastic Leukemia correctly not detected (True Negative)"
				elif response["Diagnosis"] == "Negative" and "_1." in data:
					fn += 1
					msg = "Acute Lymphoblastic Leukemia incorrectly not detected (False Negative)"

				files += 1

				self.Helpers.logger.info(msg)
				time.sleep(7)

		self.Helpers.logger.info("Images Classifier: " + str(files))
		self.Helpers.logger.info("True Positives: " + str(tp))
		self.Helpers.logger.info("False Positives: " + str(fp))
		self.Helpers.logger.info("True Negatives: " + str(tn))
		self.Helpers.logger.info("False Negatives: " + str(fn))

	def http_classify(self, req):
		""" Classifies an image sent via HTTP. """

		if len(req.files) != 0:
			img = Image.open(req.files['file'].stream)
		else:
			img = Image.open(BytesIO(req.data))

		img = self.reshape(img)

		return self.get_predictions(img)

	def get_predictions(self, img):
		""" Gets a prediction for an image. """

		predictions = self.net.infer(inputs={self.input_blob: img})
		predictions = predictions[list(predictions.keys())[0]]
		prediction = np.argsort(predictions[0])[-1]

		return prediction

	def reshape(self, img):
		""" Processes the image. """

		n, c, h, w = [1, 3, self.Helpers.confs["cnn"]["data"]["dim"],
					self.Helpers.confs["cnn"]["data"]["dim"]]
		processed = img.resize((h, w), resample = Image.BILINEAR)
		processed = (np.array(processed) - 0) / 255.0
		processed = processed.transpose((2, 0, 1))
		processed = processed.reshape((n, c, h, w))

		return processed
