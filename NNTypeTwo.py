from Results import Results
from PrepImage import PrepImage

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import LSTM, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop, SGD
# Import the backend
from keras import backend as K
import keras

class NNTypeTwo:
	def __init__(self, verbose=0):
		self.verbose = verbose
		self.model = None
		self.history = None

	def create(self, X_flat_train, num_unique_labels):
		self.model = Sequential()

		# Add dense layers to create a fully connected MLP
		# Note that we specify an input shape for the first layer, but only the first layer.
		# Relu is the activation function used
		self.model.add(Dense(256, activation='relu', input_shape=(X_flat_train.shape[1],)))
		# Dropout layers remove features and fight overfitting
		self.model.add(Dropout(0.05))
		self.model.add(Dense(128, activation='relu'))
		self.model.add(Dropout(0.05))
		self.model.add(Dense(128, activation='relu'))
		self.model.add(Dropout(0.05))
		self.model.add(Dense(128, activation='relu'))
		self.model.add(Dropout(0.05))
		self.model.add(Dense(128, activation='relu'))
		self.model.add(Dropout(0.05))
		# End with a number of units equal to the number of classes we have for our outcome
		self.model.add(Dense(num_unique_labels, activation='softmax'))

		self.model.summary()

		# Compile the model to put it all together.
		self.model.compile(loss='categorical_crossentropy',
		              optimizer=RMSprop(),
		              metrics=['accuracy'])

	def train_and_evaluate(self, X_flat_train, X_flat_test, Y_train, Y_test, epochs=10):
		self.history = self.model.fit(X_flat_train, Y_train,
	                      batch_size=128,
	                      epochs=10,
	                      verbose=self.verbose,
	                      validation_data=(X_flat_test, Y_test))

		score = self.model.evaluate(X_flat_test, Y_test, verbose=self.verbose)
		print("NNTypeTwo | Evaluating NN: Test loss={:.3f}, Test accuracy={:.3f}".format(score[0], score[1]))
		return score

	def predict(self, img_dir, id_to_label):
		process_imgs, test_images = PrepImage.prep_test_data(img_dir)
		predictions = self.model.predict_classes(process_imgs,verbose=self.verbose)
		Results.show(test_images, predictions=predictions, id_to_label=id_to_label)
		# print(predictions)


