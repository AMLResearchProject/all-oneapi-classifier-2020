############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    oneAPI Acute Lymphoblastic Leukemia Classifier
# Project:       OneAPI OpenVINO UP2 Acute Lymphoblastic Leukemia Classifier
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         ALLOpenVINO UP2
# Description:   Core class for the OneAPI OpenVINO UP2 Acute Lymphoblastic Leukemia Classifier.
# License:       MIT License
# Last Modified: 2020-09-04
#
############################################################################################

import sys

from Classes.Helpers import Helpers
from Classes.OpenVINO import OpenVINO
from Classes.Server import Server

class ALLOpenVINO():
	""" ALLOpenVINO UP2

	Core class for the OneAPI OpenVINO UP2 Acute Lymphoblastic Leukemia Classifier.
	"""

	def __init__(self):
		""" Initializes the class. """

		self.Helpers = Helpers("Core")
		self.Core = OpenVINO()

		self.Helpers.logger.info("Class initialization complete.")

	def do_classify(self):
		""" Loads model and classifies test data """

		self.Core.do_model()
		self.Core.test_classifier()

	def do_server(self):
		""" Loads the API server """

		self.Core.do_model()
		self.Server = Server(self.Core)
		self.Server.start()

	def do_http_classify(self):
		""" Classifies test data """

		self.Core.test_http_classifier()

ALLOpenVINO = ALLOpenVINO()

def main():

	if len(sys.argv) < 2:
		print("You must provide an argument")
		exit()
	elif sys.argv[1] not in ALLOpenVINO.Helpers.confs["cnn"]["core"]:
		print("Mode not supported! Server, Train or Classify")
		exit()

	mode = sys.argv[1]

	if mode == "Classify":
		""" Runs the classifier locally."""
		ALLOpenVINO.do_classify()

	elif mode == "Server":
		""" Runs the classifier in server mode."""
		ALLOpenVINO.do_server()

	elif mode == "Client":
		""" Runs the classifier in client mode. """
		ALLOpenVINO.do_http_classify()

if __name__ == "__main__":
	main()
