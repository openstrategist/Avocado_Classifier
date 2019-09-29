from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import LSTM, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop, SGD
# Import the backend
from keras import backend as K

class CreateModel:
	@staticmethod
	def create(X_flat_train, unique_labels):
		# Run a Quick Model¶
		# print("********************************************")
		# print("")
		# print("****************** TEST 1 ******************")
		# print("")
		# print("********************************************")

		model_dense = Sequential()

		# Add dense layers to create a fully connected MLP
		# Note that we specify an input shape for the first layer, but only the first layer.
		# Relu is the activation function used
		model_dense.add(Dense(128, activation='relu', input_shape=(X_flat_train.shape[1],)))
		# Dropout layers remove features and fight overfitting
		model_dense.add(Dropout(0.1))
		model_dense.add(Dense(64, activation='relu'))
		model_dense.add(Dropout(0.1))
		# End with a number of units equal to the number of classes we have for our outcome
		model_dense.add(Dense(len(unique_labels), activation='softmax'))

		model_dense.summary()

		# Compile the model to put it all together.
		model_dense.compile(loss='categorical_crossentropy',
		              optimizer=RMSprop(),
		              metrics=['accuracy'])

		return model_dense