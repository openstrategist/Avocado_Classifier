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

class NNTypeThree:
	def __init__(self, verbose=0):
		self.verbose = verbose
		self.model = None
		self.history = None

	def create(self, X_flat_train, num_unique_labels):
		self.model = Sequential()
		# First convolutional layer, note the specification of shape
		self.model.add(Conv2D(32, kernel_size=(3, 3),
		                 activation='relu',
		                 input_shape=(45, 45, 3)))
		self.model.add(Conv2D(64, (3, 3), activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))
		self.model.add(Flatten())
		self.model.add(Dense(128, activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(num_unique_labels, activation='softmax'))

		self.model.compile(loss=keras.losses.categorical_crossentropy,
		              optimizer=keras.optimizers.Adadelta(),
		              metrics=['accuracy'])

	def train_and_evaluate(self, X_train, X_test, Y_train, Y_test, epochs=10):
		history_dense = self.model.fit(X_train, Y_train,
		                          batch_size=128,
		                          epochs=epochs,
		                          verbose=self.verbose,
		                          validation_data=(X_test, Y_test))

		score = self.model.evaluate(X_test, Y_test, verbose=self.verbose)
		print("NNTypeThree | Evaluating NN: Test loss={:.3f}, Test accuracy={:.3f}".format(score[0], score[1]))
		return score

	def predict(self, img_dir, id_to_label):
		process_imgs, test_images = PrepImage.prep_test_data(img_dir)
		predictions = self.model.predict_classes(test_images,verbose=self.verbose)
		Results.show(test_images, predictions=predictions, id_to_label=id_to_label)
		# print(predictions)


