from PrepImage import PrepImage
from CreateModel import CreateModel
from Results import Results

# CONSTANTS DEFINITION
VERBOSE = 0	# {0, 1}

# A: Setting up Images & Labels
training_images, training_labels, unique_labels, id_to_label_dict = PrepImage.prep_training_data("./input/training/*")
validation_images, validation_labels = PrepImage.prep_validation_data("./input/validation/*")
X_flat_train, X_flat_test, Y_train, Y_test = PrepImage.post_process(training_images, training_labels, validation_images, validation_labels, unique_labels)
print("1) Imported training and validation data. Unique labels={}, X_flat_train size={}".format(len(unique_labels), X_flat_train.shape))

# B: Set up and Train NN
print("2) Creating and Training NN")
model_dense = CreateModel.create(X_flat_train, unique_labels)
history_dense = model_dense.fit(X_flat_train, Y_train,
                          batch_size=128,
                          epochs=10,
                          verbose=VERBOSE,
                          validation_data=(X_flat_test, Y_test))

# C: Evaluate Accuracy of NN
score = model_dense.evaluate(X_flat_test, Y_test, verbose=VERBOSE)
print("3) Evaluating NN: Test loss={}, Test accuracy={}".format(score[0], score[1]))

# D: Predict using NN
print("4) Running Prediction")
process_imgs, test_images = PrepImage.prep_test_data("./input/test")
predictions = model_dense.predict_classes(process_imgs,verbose=VERBOSE)
Results.show(predictions, test_images, id_to_label_dict)
print(predictions)








# # Deeper NN
# print("********************************************")
# print("")
# print("****************** TEST 2 ******************")
# print("")
# print("********************************************")
#
# model_deep = Sequential()
#
# # Add dense layers to create a fully connected MLP
# # Note that we specify an input shape for the first layer, but only the first layer.
# # Relu is the activation function used
# model_deep.add(Dense(256, activation='relu', input_shape=(X_flat_train.shape[1],)))
# # Dropout layers remove features and fight overfitting
# model_deep.add(Dropout(0.05))
# model_deep.add(Dense(128, activation='relu'))
# model_deep.add(Dropout(0.05))
# model_deep.add(Dense(128, activation='relu'))
# model_deep.add(Dropout(0.05))
# model_deep.add(Dense(128, activation='relu'))
# model_deep.add(Dropout(0.05))
# model_deep.add(Dense(128, activation='relu'))
# model_deep.add(Dropout(0.05))
# # End with a number of units equal to the number of classes we have for our outcome
# model_deep.add(Dense(60, activation='softmax'))
#
# model_deep.summary()
#
# # Compile the model to put it all together.
# model_deep.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])
#
# history_deep = model_deep.fit(X_flat_train, Y_train,
#                           batch_size=128,
#                           epochs=10,
#                           verbose=1,
#                           validation_data=(X_flat_test, Y_test))
# score = model_deep.evaluate(X_flat_test, Y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
#
# # Even deeper
# print("********************************************")
# print("")
# print("****************** TEST 3 ******************")
# print("")
# print("********************************************")
#
# model_cnn = Sequential()
# # First convolutional layer, note the specification of shape
# model_cnn.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=(45, 45, 3)))
# model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
# model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
# model_cnn.add(Dropout(0.25))
# model_cnn.add(Flatten())
# model_cnn.add(Dense(128, activation='relu'))
# model_cnn.add(Dropout(0.5))
# model_cnn.add(Dense(60, activation='softmax'))
#
# model_cnn.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#
# model_cnn.fit(X_train, Y_train,
#           batch_size=128,
#           epochs=1,
#           verbose=1,
#           validation_data=(X_test, Y_test))
# score = model_cnn.evaluate(X_test, Y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
# # Last run
#
# print("********************************************")
# print("")
# print("****************** TEST 4 ******************")
# print("")
# print("********************************************")
#
# model_cnn.fit(X_train, Y_train,
#           batch_size=128,
#           epochs=10,
#           verbose=1,
#           validation_data=(X_test, Y_test))
# score = model_cnn.evaluate(X_test, Y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
