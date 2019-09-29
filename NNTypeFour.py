# Reference: https://www.kaggle.com/suniliitb96/tutorial-keras-transfer-learning-with-resnet50

from Results import Results
from PrepImage import PrepImage

# from keras.models import Sequential
# from keras.optimizers import RMSprop, SGD
# from keras.layers import Input, Conv2D, MaxPooling2D
# from keras.layers import Dense, Flatten
# from keras.models import Model
# from keras.preprocessing import image

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import optimizers
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
# Early stopping & checkpointing the best model in ../working dir & restoring that as our model for prediction
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import the backend
from keras import backend as K
import keras

import matplotlib.pyplot as plt

class NNTypeFour:
	def __init__(self, verbose=0):
		self.verbose = verbose
		self.model = None
		self.history = None

	def create(self, X_flat_train, num_unique_labels):
		resnet_weights_path = './transfer_learning_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

		model = Sequential()

		# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
		# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
		model.add(ResNet50(include_top = False, pooling = 'avg', weights = resnet_weights_path))

		# 2nd layer as Dense for (num_unique_labels)-class classification, i.e., dog or cat using SoftMax activation
		model.add(Dense(num_unique_labels, activation = 'softmax'))

		# Say not to train first layer (ResNet) model as it is already trained
		model.layers[0].trainable = False

		model.summary()

		sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
		model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

		self.model = model

	def train_and_evaluate(self, X_train, X_test, Y_train, Y_test, epochs=10):
		train_img_dir = './input/training'
		validate_img_dir = './input/validation'

		#image_size = 224
		image_size = 45

		# preprocessing_function is applied on each image but only after re-sizing & augmentation (resize => augment => pre-process)
		# Each of the keras.application.resnet* preprocess_input MOSTLY mean BATCH NORMALIZATION (applied on each batch) stabilize the inputs to nonlinear activation functions
		# Batch Normalization helps in faster convergence
		data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

		# flow_From_directory generates batches of augmented data (where augmentation can be color conversion, etc)
		# Both train & valid folders must have NUM_CLASSES sub-folders
		train_generator = data_generator.flow_from_directory(
				train_img_dir,
		        target_size=(image_size, image_size),
		        batch_size=100,
		        class_mode='categorical')

		validation_generator = data_generator.flow_from_directory(
				validate_img_dir,
				target_size=(image_size, image_size),
		        batch_size=100,
		        class_mode='categorical') 

		# cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
		# cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')

		fit_history = self.model.fit_generator(
		        train_generator,
		        steps_per_epoch=10,
		        epochs=10,
		        validation_data=validation_generator,
		        validation_steps=10
		        # callbacks=[cb_checkpointer, cb_early_stopper]
		)

		# model.load_weights("../working/best.hdf5")

		# score = self.model.evaluate(imgs, Y_test, verbose=self.verbose)

		plt.figure(1, figsize = (15,8)) 
		    
		plt.subplot(221)  
		plt.plot(fit_history.history['acc'])  
		plt.plot(fit_history.history['val_acc'])  
		plt.title('model accuracy')  
		plt.ylabel('accuracy')  
		plt.xlabel('epoch')  
		plt.legend(['train', 'valid']) 

		plt.subplot(222)
		plt.plot(fit_history.history['loss'])  
		plt.plot(fit_history.history['val_loss'])
		plt.title('model loss')  
		plt.ylabel('loss')  
		plt.xlabel('epoch')  
		plt.legend(['train', 'valid']) 

		plt.show()

		# print("history_keys={}".format(fit_history.history.keys()))
		# print("history_history={}".format(fit_history.history))
		# acc = fit_history.history['acc']
		# loss = fit_history.history['loss']

		# print("NNTypeFour | Evaluating NN: Test loss={:.3f}, Test accuracy={:.3f}".format(acc, loss))
		# return [acc, loss]

	def predict(self, img_dir, id_to_label):
		# process_imgs, test_images = PrepImage.prep_test_data(img_dir)
		# predictions = self.model.predict_classes(process_imgs,verbose=self.verbose)
		# Results.show(test_images, predictions=predictions, id_to_label=id_to_label)
		# # print(predictions)
		None
