############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    oneAPI Acute Lymphoblastic Leukemia Classifier
# Project:       OneAPI OpenVINO Raspberry Pi 4 Acute Lymphoblastic Leukemia Classifier
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
#
# Title:         Server Class
# Description:   Server class for the OneAPI OpenVINO Raspberry Pi 4 Acute Lymphoblastic
#                Leukemia Classifier.
# License:       MIT License
# Last Modified: 2020-09-04
#
############################################################################################

import jsonpickle

from flask import Flask, request, Response

from Classes.Helpers import Helpers

class Server():
	""" Server helper class

	Server functions for the OneAPI OpenVINO Raspberry Pi 4 Acute
	Lymphoblastic Leukemia Classifier.
	"""

	def __init__(self, model):
		""" Initializes the class. """

		self.Helpers = Helpers("Server", False)

		self.model = model

		self.Helpers.logger.info("Class initialization complete.")

	def start(self):
		""" Starts the server. """

		app = Flask(__name__)
		@app.route('/Inference', methods=['POST'])
		def Inference():
			""" Responds to standard HTTP request. """

			message = ""
			classification = self.model.http_classify(request)

			if classification == 1:
				message = "Acute Lymphoblastic Leukemia detected!"
				diagnosis = "Positive"
			elif classification == 0:
				message = "Acute Lymphoblastic Leukemia not detected!"
				diagnosis = "Negative"

			resp = jsonpickle.encode({
				'Response': 'OK',
				'Message': message,
				'Diagnosis': diagnosis
			})

			return Response(response=resp, status=200, mimetype="application/json")

		app.run(host = self.Helpers.confs["cnn"]["system"]["server"],
				port = self.Helpers.confs["cnn"]["system"]["port"])
