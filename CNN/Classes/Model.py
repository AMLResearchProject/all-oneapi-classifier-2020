############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    OneAPI Acute Lymphoblastic Leukemia Classifier
# Project:       ALLoneAPI CNN
#
# Author:        Adam Milton-Barker (AdamMiltonBarker.com)
# Contributors:
# Title:         Model Class
# Description:   Model functions for the OneAPI Acute Lymphoblastic Leukemia Classifier.
# License:       MIT License
# Last Modified: 2020-09-03
#
############################################################################################

import cv2
import json
import os
import random
import requests
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from numpy.random import seed
from tensorflow.keras import layers, models
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from sklearn.metrics import confusion_matrix

from Classes.Helpers import Helpers
from Classes.Data import Data

class Model():
	""" Model Class

	Model functions for the OneAPI Acute Lymphoblastic Leukemia Classifier.
	"""

	def __init__(self):
		""" Initializes the class. """

		self.Helpers = Helpers("Model", False)

		os.environ["KMP_BLOCKTIME"] = "1"
		os.environ["KMP_SETTINGS"] = "1"
		os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
		os.environ["OMP_NUM_THREADS"] = str(self.Helpers.confs["cnn"]["system"]["cores"])
		tf.config.threading.set_inter_op_parallelism_threads(1)
		tf.config.threading.set_intra_op_parallelism_threads(
			self.Helpers.confs["cnn"]["system"]["cores"])

		self.testing_dir = self.Helpers.confs["cnn"]["data"]["test"]
		self.valid = self.Helpers.confs["cnn"]["data"]["valid_types"]
		self.seed = self.Helpers.confs["cnn"]["data"]["seed"]

		self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

		self.weights_file = self.Helpers.confs["cnn"]["model"]["weights"]
		self.model_json = self.Helpers.confs["cnn"]["model"]["model"]

		random.seed(self.seed)
		seed(self.seed)
		tf.random.set_seed(self.seed)

		self.Helpers.logger.info("Model class initialization complete.")

	def do_data(self):
		""" Creates/sorts dataset. """

		self.Data = Data()
		self.Data.do_im_process()

		self.Helpers.logger.info("Data preperation complete.")

	def do_network(self):
		""" Builds the network.

		Replicates the networked outlined in the  Acute Leukemia Classification
		Using Convolution Neural Network In Clinical Decision Support System paper
		using Tensorflow 2.0.
		https://airccj.org/CSCP/vol7/csit77505.pdf
		"""

		self.val_steps = self.Helpers.confs["cnn"]["train"]["val_steps"]
		self.batch_size = self.Helpers.confs["cnn"]["train"]["batch"]
		self.epochs = self.Helpers.confs["cnn"]["train"]["epochs"]

		self.tf_model = tf.keras.models.Sequential([
			tf.keras.layers.ZeroPadding2D(
				padding=(2, 2), input_shape=self.Data.X_train.shape[1:]),
			tf.keras.layers.Conv2D(30, (5, 5), strides=1,
								   padding="valid", activation='relu'),
			tf.keras.layers.ZeroPadding2D(padding=(2, 2)),
			tf.keras.layers.Conv2D(30, (5, 5), strides=1,
								   padding="valid", activation='relu'),
			tf.keras.layers.MaxPooling2D(
				pool_size=(2, 2), strides=2, padding='valid'),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(2),
			tf.keras.layers.Activation('softmax')
		],
		"AllDS2020_TF_CNN")
		self.tf_model.summary()
		self.Helpers.logger.info("Network initialization complete.")

	def do_train(self):
		""" Trains the network. """

		self.Helpers.logger.info("Using Adam Optimizer.")
		optimizer = tf.keras.optimizers.Adam(lr=self.Helpers.confs["cnn"]["train"]["learning_rate_adam"],
												decay = self.Helpers.confs["cnn"]["train"]["decay_adam"])

		self.tf_model.compile(optimizer=optimizer,
						   loss='binary_crossentropy',
						   metrics=[tf.keras.metrics.BinaryAccuracy(name='acc'),
									tf.keras.metrics.Precision(name='precision'),
									tf.keras.metrics.Recall(name='recall'),
									tf.keras.metrics.AUC(name='auc') ])

		self.history = self.tf_model.fit(self.Data.X_train, self.Data.y_train, validation_data=(self.Data.X_test, self.Data.y_test),
									  validation_steps=self.val_steps, epochs=self.epochs)

		print(self.history)
		print("")

		self.freeze_model()
		self.save_model_as_json()
		self.save_weights()

	def do_evaluate(self):
		""" Evaluates the model """

		self.do_predictions()

		metrics = self.tf_model.evaluate(self.Data.X_test, self.Data.y_test, verbose=0)
		for name, value in zip(self.tf_model.metrics_names, metrics):
			self.Helpers.logger.info("Metrics: " + name + " " + str(value))
		print()

		self.visualize_metrics()
		self.confusion_matrix()
		self.figures_of_merit()

	def do_predictions(self):
		""" Makes predictions on the train & test sets. """

		self.train_preds = self.tf_model.predict(self.Data.X_train)
		self.test_preds = self.tf_model.predict(self.Data.X_test)

		self.Helpers.logger.info("Training predictions: " + str(self.train_preds))
		self.Helpers.logger.info("Testing predictions: " + str(self.test_preds))
		print("")

	def visualize_metrics(self):
		""" Visualize the metrics. """

		plt.plot(self.history.history['acc'])
		plt.plot(self.history.history['val_acc'])
		plt.title('Model Accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.ylim((0, 1))
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('Model/Plots/Accuracy.png')
		plt.show()
		plt.clf()

		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('Model Loss')
		plt.ylabel('loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('Model/Plots/Loss.png')
		plt.show()
		plt.clf()

		plt.plot(self.history.history['auc'])
		plt.plot(self.history.history['val_auc'])
		plt.title('Model AUC')
		plt.ylabel('AUC')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('Model/Plots/AUC.png')
		plt.show()
		plt.clf()

		plt.plot(self.history.history['precision'])
		plt.plot(self.history.history['val_precision'])
		plt.title('Model Precision')
		plt.ylabel('Precision')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('Model/Plots/Precision.png')
		plt.show()
		plt.clf()

		plt.plot(self.history.history['recall'])
		plt.plot(self.history.history['val_recall'])
		plt.title('Model Recall')
		plt.ylabel('Recall')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('Model/Plots/Recall.png')
		plt.show()
		plt.clf()

	def confusion_matrix(self):
		""" Prints/displays the confusion matrix. """

		self.matrix = confusion_matrix(self.Data.y_test.argmax(axis=1),
									   self.test_preds.argmax(axis=1))

		self.Helpers.logger.info("Confusion Matrix: " + str(self.matrix))
		print("")

		plt.imshow(self.matrix, cmap=plt.cm.Blues)
		plt.xlabel("Predicted labels")
		plt.ylabel("True labels")
		plt.xticks([], [])
		plt.yticks([], [])
		plt.title('Confusion matrix ')
		plt.colorbar()
		plt.savefig('Model/Plots/Confusion-Matrix.png')
		plt.show()
		plt.clf()

	def figures_of_merit(self):
		""" Calculates/prints the figures of merit.

		https://homes.di.unimi.it/scotti/all/
		"""

		test_len = len(self.Data.X_test)

		TP = self.matrix[1][1]
		TN = self.matrix[0][0]
		FP = self.matrix[0][1]
		FN = self.matrix[1][0]

		TPP = (TP * 100)/test_len
		FPP = (FP * 100)/test_len
		FNP = (FN * 100)/test_len
		TNP = (TN * 100)/test_len

		specificity = TN/(TN+FP)

		misc = FP + FN
		miscp = (misc * 100)/test_len

		self.Helpers.logger.info("True Positives: " + str(TP) + "(" + str(TPP) + "%)")
		self.Helpers.logger.info("False Positives: " + str(FP) + "(" + str(FPP) + "%)")
		self.Helpers.logger.info("True Negatives: " + str(TN) + "(" + str(TNP) + "%)")
		self.Helpers.logger.info("False Negatives: " + str(FN) + "(" + str(FNP) + "%)")

		self.Helpers.logger.info("Specificity: " + str(specificity))
		self.Helpers.logger.info("Misclassification: " +
								str(misc) + "(" + str(miscp) + "%)")

	def freeze_model(self):
		""" Saves the model weights. """

		tf.saved_model.save(
			self.tf_model, self.Helpers.confs["cnn"]["model"]["saved_model_dir"])

		fmodel = tf.function(lambda x: self.tf_model(x))
		fmodel = fmodel.get_concrete_function(
			x=tf.TensorSpec(self.tf_model.inputs[0].shape, self.tf_model.inputs[0].dtype))

		freeze = convert_variables_to_constants_v2(fmodel)
		freeze.graph.as_graph_def()

		layers = [op.name for op in freeze.graph.get_operations()]
		self.Helpers.logger.info("Frozen model layers")
		for layer in layers:
			self.Helpers.logger.info(layer)

		self.Helpers.logger.info("Frozen model inputs")
		self.Helpers.logger.info(freeze.inputs)
		self.Helpers.logger.info("Frozen model outputs")
		self.Helpers.logger.info(freeze.outputs)

		tf.io.write_graph(graph_or_graph_def=freeze.graph,
			logdir=self.Helpers.confs["cnn"]["model"]["freezing_log_dir"],
			name=self.Helpers.confs["cnn"]["model"]["frozen"],
			as_text=False)

	def save_weights(self):
		""" Saves the model weights. """

		self.tf_model.save_weights(self.weights_file)
		self.Helpers.logger.info("Weights saved " + self.weights_file)

	def save_model_as_json(self):
		""" Saves the model to JSON. """

		with open(self.model_json, "w") as file:
			file.write(self.tf_model.to_json())

		self.Helpers.logger.info("Model JSON saved " + self.model_json)

	def load_model_and_weights(self):
		""" Loads the model and weights. """

		with open(self.model_json) as file:
			m_json = file.read()

		self.tf_model = tf.keras.models.model_from_json(m_json)
		self.tf_model.load_weights(self.weights_file)

		self.Helpers.logger.info("Model loaded ")

		self.tf_model.summary()

	def test_classifier(self):
		""" Tests the trained model. """

		files = 0
		tp = 0
		fp = 0
		tn = 0
		fn = 0

		for testFile in os.listdir(self.testing_dir):
			if os.path.splitext(testFile)[1] in self.valid:

				files += 1
				fileName = self.testing_dir + "/" + testFile

				img = cv2.imread(fileName).astype(np.float32)
				self.Helpers.logger.info("Loaded test image " + fileName)

				img = cv2.resize(img, (self.Helpers.confs["cnn"]["data"]["dim"],
									   self.Helpers.confs["cnn"]["data"]["dim"]))
				img = self.reshape(img)

				prediction = self.get_predictions(img)

				msg = ""
				if prediction == 1 and "_1." in testFile:
					tp += 1
					msg = "Acute Lymphoblastic Leukemia correctly detected (True Positive)"
				elif prediction == 1 and "_0." in testFile:
					fp += 1
					msg = "Acute Lymphoblastic Leukemia incorrectly detected (False Positive)"
				elif prediction == 0 and "_0." in testFile:
					tn += 1
					msg = "Acute Lymphoblastic Leukemia correctly not detected (True Negative)"
				elif prediction == 0 and "_1." in testFile:
					fn += 1
					msg = "Acute Lymphoblastic Leukemia incorrectly not detected (False Negative)"
				self.Helpers.logger.info(msg)

		self.Helpers.logger.info("Images Classifier: " + str(files))
		self.Helpers.logger.info("True Positives: " + str(tp))
		self.Helpers.logger.info("False Positives: " + str(fp))
		self.Helpers.logger.info("True Negatives: " + str(tn))
		self.Helpers.logger.info("False Negatives: " + str(fn))

	def send_request(self, img_path):
		""" Sends image to the inference API endpoint. """

		self.Helpers.logger.info("Sending request for: " + img_path)

		_, img_encoded = cv2.imencode('.png', cv2.imread(img_path))
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
				print()
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
		img = cv2.resize(img, (self.Helpers.confs["cnn"]["data"]["dim"],
							   self.Helpers.confs["cnn"]["data"]["dim"]))
		img = self.reshape(img)

		return self.get_predictions(img)

	def vr_http_classify(self, img):
		""" Classifies an image sent via from VR via HTTP. """

		img = cv2.resize(img, (self.Helpers.confs["cnn"]["data"]["dim"],
							   self.Helpers.confs["cnn"]["data"]["dim"]))
		img = self.reshape(img)

		return self.get_predictions(img)

	def get_predictions(self, img):
		""" Gets a prediction for an image. """

		predictions = self.tf_model.predict_proba(img)
		prediction = np.argmax(predictions, axis=-1)

		return prediction

	def reshape(self, img):
		""" Reshapes an image. """

		dx, dy, dz = img.shape
		input_data = img.reshape((-1, dx, dy, dz))

		return input_data
