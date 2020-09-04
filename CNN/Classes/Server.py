############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    OneAPI Acute Lymphoblastic Leukemia Classifier
# Project:       ALLoneAPI CNN
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Server helper class
# Description:   Server functions for the OneAPI Acute Lymphoblastic Leukemia Classifier.
# License:       MIT License
# Last Modified: 2020-09-03
#
############################################################################################
import jsonpickle

from flask import Flask, request, Response

from Classes.Helpers import Helpers

class Server():
	""" Server helper class

	Server functions for the OneAPI Acute Lymphoblastic Leukemia Classifier.
	"""

	def __init__(self, model):
		""" Initializes the class. """

		self.Helpers = Helpers("Server", False)

		self.model = model

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

		@app.route('/VRInference', methods=['POST'])
		def VRInference(id):
			""" Responds to requests from Oculus Rift. """

			t_drive = self.Helpers.confs["cnn"]["data"]["test_data"]

			if int(id)-1 > len(t_drive):
				ServerResponse = jsonpickle.encode({
					'Response': 'FAILED',
					'Message': 'No testing data with provided ID'
				})

			i = int(id)-1

			test_image = self.Helpers.confs["cnn"]["data"]["test"] + "/" + t_drive[i]

			if not os.path.isfile(test_image):
				ServerResponse = jsonpickle.encode({
					'Response': 'FAILED',
					'Message': 'No testing data with filename exists'
				})

			message = ""
			classification = self.model.vr_http_classify(cv2.imread(test_image))

			if classification == 1:
				msg = "Positive"
			elif classification == 0:
				message  = "Negative"

			resp = jsonpickle.encode({
				'Response': 'OK',
				'Message': message,
				'Classification': classification
			})

			return Response(response=resp, status=200, mimetype="application/json")

		app.run(host = self.Helpers.confs["cnn"]["system"]["server"],
				port = self.Helpers.confs["cnn"]["system"]["port"])
