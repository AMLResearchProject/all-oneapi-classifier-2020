#########################################################################################################
#
# Organization:     Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss
# Research Project: Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:       oneAPI Acute Lymphoblastic Leukemia Classifier
# Project:          OneAPI OpenVINO Raspberry Pi 4 Acute Lymphoblastic Leukemia Classifier
#
# Author:           Adam Milton-Barker (AdamMiltonBarker.com)
#
# Title:            OpenVINO Class
# Description:      OpenVINO class for the OneAPI OpenVINO Raspberry Pi 4 Acute Lymphoblastic
#                   Leukemia Classifier.
# License:          MIT License
# Last Modified:    2020-10-02
#
#########################################################################################################

import cv2
import json
import os
import requests
import time

import numpy as np

from io import BytesIO
from PIL import Image

from Classes.Helpers import Helpers

class OpenVINO():
	""" OpenVINO Class

	OpenVINO class for the OneAPI OpenVINO Raspberry Pi 4 Acute
	Lymphoblastic Leukemia Classifier.
	"""

	def __init__(self):
		""" Initializes the Model class. """

		self.Helpers = Helpers("OpenVINO")

		self.testing_dir = self.Helpers.confs["cnn"]["data"]["test"]
		self.valid = self.Helpers.confs["cnn"]["data"]["valid_types"]

		mxml = self.Helpers.confs["cnn"]["rpi4"]["ir"]
		mbin = os.path.splitext(mxml)[0] + ".bin"

		self.net = cv2.dnn.readNet(mxml, mbin)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

		self.imsize = self.Helpers.confs["cnn"]["data"]["dim"]

		self.Helpers.logger.info("Class initialization complete.")

	def setBlob(self, frame):
		""" Gets a blob from the color frame """

		blob = cv2.dnn.blobFromImage(frame, self.Helpers.confs["cnn"]["rpi4"]["inScaleFactor"],
									size=(self.imsize, self.imsize),
									mean=(self.Helpers.confs["cnn"]["rpi4"]["meanVal"],
										self.Helpers.confs["cnn"]["rpi4"]["meanVal"],
										self.Helpers.confs["cnn"]["rpi4"]["meanVal"]),
									swapRB=True, crop=False)

		self.net.setInput(blob)

	def forwardPass(self):
		""" Gets a blob from the color frame """

		out = self.net.forward()

		return out

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
				img = cv2.imread(fileName)
				self.Helpers.logger.info("Loaded test image " + fileName)
				img = self.reshape(img)
				self.setBlob(img)
				prediction = self.get_predictions()
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
			img = np.fromstring(req.files['file'].read(), np.uint8)
		else:
			img = np.fromstring(req.data, np.uint8)

		img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

		img = self.reshape(img)
		self.setBlob(img)

		return self.get_predictions()

	def get_predictions(self):
		""" Gets a prediction for an image. """

		predictions = self.forwardPass()
		predictions = predictions[0]
		idx = np.argsort(predictions)[::-1][0]
		prediction = self.Helpers.confs["cnn"]["data"]["labels"][idx]

		return prediction

	def reshape(self, img):
		""" Reshapes an image. """

		img = cv2.resize(img, (self.Helpers.confs["cnn"]["data"]["dim"],
								self.Helpers.confs["cnn"]["data"]["dim"]))

		return img
