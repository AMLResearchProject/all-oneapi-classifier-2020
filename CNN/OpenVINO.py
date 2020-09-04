############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    OneAPI Acute Lymphoblastic Leukemia Classifier
# Project:       ALLoneAPI CNN
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         OpenVINO Class
# Description:   OpenVINO functions for the OneAPI Acute Lymphoblastic Leukemia Classifier.
# License:       MIT License
# Last Modified: 2020-09-03
#
############################################################################################

import os

from openvino import inference_engine as ie
from openvino.inference_engine import IENetwork, IECore

from PIL import Image
import numpy as np

from Classes.Helpers import Helpers

class OpenVINO():
	""" OpenVINO Class

	OpenVINO functions for the OneAPI Acute Lymphoblastic Leukemia Classifier.
	"""

	def __init__(self):
		""" Initializes the class. """

		self.Helpers = Helpers("OpenVINO")

		os.environ["KMP_BLOCKTIME"] = "1"
		os.environ["KMP_SETTINGS"] = "1"
		os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
		os.environ["OMP_NUM_THREADS"] = str(
			self.Helpers.confs["cnn"]["system"]["cores"])

		self.testing_dir = self.Helpers.confs["cnn"]["data"]["test"]
		self.valid = self.Helpers.confs["cnn"]["data"]["valid_types"]

		self.Helpers.logger.info("ALLoneAPI OpenVINO initialization complete.")

	def im_process(self, imagePath):
		""" Processes the image. """

		n, c, h, w = [1, 3, self.Helpers.confs["cnn"]["data"]["dim"],
					self.Helpers.confs["cnn"]["data"]["dim"]]
		image = Image.open(imagePath)
		processed = image.resize((h, w), resample = Image.BILINEAR)
		processed = (np.array(processed) - 0) / 255.0
		processed = processed.transpose((2, 0, 1))
		processed = processed.reshape((n, c, h, w))

		return processed

	def test(self):
		""" Tests the OpenVINO model. """

		mxml = self.Helpers.confs["cnn"]["model"]["ir"]
		mbin = os.path.splitext(mxml)[0] + ".bin"

		ie = IECore()
		net = ie.read_network(model = mxml, weights = mbin)
		input_blob = next(iter(net.inputs))
		net = ie.load_network(network=net,
						device_name=self.Helpers.confs["cnn"]["model"]["device"])

		files = 0
		tp = 0
		fp = 0
		tn = 0
		fn = 0

		for testFile in os.listdir(self.testing_dir):
			if os.path.splitext(testFile)[1] in self.valid:

				files += 1
				fileName = self.testing_dir + "/" + testFile
				processed = self.im_process(fileName)
				res = net.infer(inputs={input_blob: processed})
				res = res[list(res.keys())[0]]
				idx = np.argsort(res[0])[-1]

				msg = ""
				if idx == 1 and "_1." in testFile:
					tp += 1
					msg = "Acute Lymphoblastic Leukemia correctly detected (True Positive)"
				elif idx == 1 and "_0." in testFile:
					fp += 1
					msg = "Acute Lymphoblastic Leukemia incorrectly detected (False Positive)"
				elif idx == 0 and "_0." in testFile:
					tn += 1
					msg = "Acute Lymphoblastic Leukemia correctly not detected (True Negative)"
				elif idx == 0 and "_1." in testFile:
					fn += 1
					msg = "Acute Lymphoblastic Leukemia incorrectly not detected (False Negative)"
				self.Helpers.logger.info(msg)

		self.Helpers.logger.info("Images Classifier: " + str(files))
		self.Helpers.logger.info("True Positives: " + str(tp))
		self.Helpers.logger.info("False Positives: " + str(fp))
		self.Helpers.logger.info("True Negatives: " + str(tn))
		self.Helpers.logger.info("False Negatives: " + str(fn))

OpenVINO = OpenVINO()
OpenVINO.test()
