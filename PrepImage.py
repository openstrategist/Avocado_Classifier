import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import glob
import cv2
import os

class PrepImage:
	@staticmethod
	def prep_training_data(training_dir):  #"./input/training/*"
		# Import images and labels.
		training_images, training_labels = [], []
		for dir_path in glob.glob(training_dir):
			label = dir_path.split("/")[-1]
			for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
				image = cv2.imread(image_path, cv2.IMREAD_COLOR)
				image = cv2.resize(image, (45, 45))
				image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

				training_images.append(image)
				training_labels.append(label)
		training_images = np.array(training_images)
		training_labels = np.array(training_labels)

		# Generate label ids
		label_to_id_dict = {v: i for i, v in enumerate(np.unique(training_labels))}
		id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
		training_label_ids = np.array([label_to_id_dict[x] for x in training_labels])
		print(label_to_id_dict)

		return training_images, training_label_ids, np.unique(training_labels), id_to_label_dict

	@staticmethod
	def prep_validation_data(validation_dir): #"./input/validation/*"
		# Validation
		validation_images, validation_labels = [], []
		for dir_path in glob.glob(validation_dir):
			label = dir_path.split("/")[-1]
			for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
				image = cv2.imread(image_path, cv2.IMREAD_COLOR)
				image = cv2.resize(image, (45, 45))
				image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

				validation_images.append(image)
				validation_labels.append(label)

		validation_images = np.array(validation_images)
		validation_labels = np.array(validation_labels)

		label_to_id_dict = {v: i for i, v in enumerate(np.unique(validation_labels))}
		validation_label_ids = np.array([label_to_id_dict[x] for x in validation_labels])

		return validation_images, validation_label_ids

	@staticmethod
	def post_process(training_images, training_label_ids, validation_images, validation_label_ids, unique_labels):
		assert(len(unique_labels) == 0, "Unique labels cannot be less than 0.")

		# Splitting the Data
		X_train, X_test = training_images, validation_images
		Y_train, Y_test = training_label_ids, validation_label_ids

		# Normalize color values to between 0 and 1
		X_train = X_train / 255
		X_test = X_test / 255

		# Make a flattened version for some of our models
		X_flat_train = X_train.reshape(X_train.shape[0], 45 * 45 * 3)  # dimension of image is 45x45.
		X_flat_test = X_test.reshape(X_test.shape[0], 45 * 45 * 3)

		# One Hot Encode the Output
		# TODO: keras.utils.to_categorical: Converts a class vector (integers) to binary class matrix.
		Y_train = keras.utils.to_categorical(Y_train, len(unique_labels))
		Y_test = keras.utils.to_categorical(Y_test, len(unique_labels))

		# print('Original Sizes:', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
		# print('Flattened:', X_flat_train.shape, X_flat_test.shape)

		print("Labels={}, Training size={}, Validation size={}".format(unique_labels, X_train.shape[0], X_test.shape[0]))

		return X_flat_train, X_flat_test, Y_train, Y_test

	# print(X_train[0].shape)
	# plt.imshow(X_train[0])
	# plt.show()

	@staticmethod
	def prep_test_data(test_dir): #"./input/test/*"
		images = []
		for image_path in glob.glob(os.path.join(test_dir, "*.jpg")):
			image = cv2.imread(image_path, cv2.IMREAD_COLOR)
			image = cv2.resize(image, (45, 45))
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

			images.append(image)
		images = np.array(images)

		process_imgs = images / 255
		process_imgs = process_imgs.reshape(process_imgs.shape[0], 45 * 45 * 3)  # dimension of image is 45x45.
		return process_imgs, images

	@staticmethod
	def get_imgs(test_dir): #"./input/test/*"
		images = []
		for image_path in glob.glob(os.path.join(test_dir, "*.jpg")):
			image = cv2.imread(image_path, cv2.IMREAD_COLOR)
			image = cv2.resize(image, (45, 45))
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

			images.append(image)
		return images