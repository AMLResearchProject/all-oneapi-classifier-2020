#########################################################################################################
#
# Organization:     Asociacion De Investigacion En Inteligencia Artificial Para La Leucemia Peter Moss
# Research Project: Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:       oneAPI Acute Lymphoblastic Leukemia Classifier
# Project:          OneAPI OpenVINO Raspberry Pi 4 Acute Lymphoblastic Leukemia Classifier
#
# Author:           Adam Milton-Barker (AdamMiltonBarker.com)
#
# Title:            ALLOpenVINO RPI 4
# Description:      Core class for the OneAPI Raspberry Pi 4 Acute Lymphoblastic Leukemia Classifier.
# License:          MIT License
# Last Modified:    2020-10-02
#
#########################################################################################################

import psutil
import signal
import requests
import sys
import threading

from threading import Thread

from Classes.Helpers import Helpers
from Classes.iotJumpWay import Device as iot
from Classes.OpenVINO import OpenVINO
from Classes.Server import Server

class ALLOpenVINO():
	""" ALLOpenVINO RPI 4

	Core class for the OneAPI OpenVINO Raspberry Pi 4 Acute Lymphoblastic Leukemia Classifier.
	"""

	def __init__(self):
		""" Initializes the class. """

		self.Helpers = Helpers("Core")
		self.Core = OpenVINO()

		self.Helpers.logger.info("Class initialization complete.")

	def iotJumpWayConn(self):

		# Initiates the iotJumpWay connection class
		self.iot = iot()
		self.iot.connect()

	def life(self):
		""" Sends vital statistics to HIAS """

		cpu = psutil.cpu_percent()
		mem = psutil.virtual_memory()[2]
		hdd = psutil.disk_usage('/').percent
		tmp = psutil.sensors_temperatures()['cpu_thermal'][0].current
		r = requests.get('http://ipinfo.io/json?token=' +
		                 self.Helpers.confs["iotJumpWay"]["ipinfo"])
		data = r.json()
		location = data["loc"].split(',')

		self.Helpers.logger.info("GeniSysAI Life (TEMPERATURE): " + str(tmp) + "\u00b0")
		self.Helpers.logger.info("GeniSysAI Life (CPU): " + str(cpu) + "%")
		self.Helpers.logger.info("GeniSysAI Life (Memory): " + str(mem) + "%")
		self.Helpers.logger.info("GeniSysAI Life (HDD): " + str(hdd) + "%")
		self.Helpers.logger.info("GeniSysAI Life (LAT): " + str(location[0]))
		self.Helpers.logger.info("GeniSysAI Life (LNG): " + str(location[1]))

		# Send iotJumpWay notification
		self.iot.channelPub("Life", {
			"CPU": str(cpu),
			"Memory": str(mem),
			"Diskspace": str(hdd),
			"Temperature": str(tmp),
			"Latitude": "",
			"Longitude": ""
		})

		threading.Timer(300.0, self.life).start()

	def do_classify(self):
		""" Loads model and classifies test data """

		self.Core.test_classifier()

	def do_server(self):
		""" Loads the API server """

		self.Server = Server(self.Core, self.iot)
		self.Server.start()

	def do_http_classify(self):
		""" Loads model and classifies test data """

		self.Core.test_http_classifier()

	def threading(self):
		""" Creates required module threads. """

		# Life thread
		threading.Timer(10.0, self.life).start()

	def signal_handler(self, signal, frame):
		self.Helpers.logger.info("Disconnecting")
		self.iot.disconnect()
		sys.exit(1)

ALLOpenVINO = ALLOpenVINO()

def main():
	# Starts threading
	signal.signal(signal.SIGINT, ALLOpenVINO.signal_handler)
	signal.signal(signal.SIGTERM, ALLOpenVINO.signal_handler)
	ALLOpenVINO.threading()

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
		ALLOpenVINO.iotJumpWayConn()
		ALLOpenVINO.do_server()

	elif mode == "Client":
		""" Runs the classifier in client mode. """
		ALLOpenVINO.do_http_classify()

if __name__ == "__main__":
	main()
