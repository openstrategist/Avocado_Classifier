from PrepImage import PrepImage
from CreateModel import CreateModel
from Results import Results
from NNTypeOne import NNTypeOne
from NNTypeTwo import NNTypeTwo
from NNTypeThree import NNTypeThree

# CONSTANTS DEFINITION
VERBOSE = 0	# {0, 1}

# 1: Setting up Images & Labels
training_images, training_labels, unique_labels, id_to_label_dict = PrepImage.prep_training_data("./input/training/*")
validation_images, validation_labels = PrepImage.prep_validation_data("./input/validation/*")
X_flat_train, X_flat_test, Y_train, Y_test = PrepImage.post_process(training_images, training_labels, validation_images, validation_labels, unique_labels)
X_train = training_images/255
X_test = validation_images/255
print("1) Imported training and validation data. Unique labels={}, X_flat_train size={}".format(len(unique_labels), X_flat_train.shape))

# 2: Run Neural Networks
# # 2A) NNTypeOne
# nnTypeOne = NNTypeOne(verbose=VERBOSE)
# nnTypeOne.create(X_flat_train, len(unique_labels))
# nnTypeOne.train_and_evaluate(X_flat_train, X_flat_test, Y_train, Y_test)
# # nnTypeOne.predict("./input/test", id_to_label_dict)	# problem: freezes program

# 2B) NNTypeTwo
# nnTypeTwo = NNTypeTwo(verbose=VERBOSE)
# nnTypeTwo.create(X_flat_train, len(unique_labels))
# nnTypeTwo.train_and_evaluate(X_flat_train, X_flat_test, Y_train, Y_test)
# nnTypeTwo.predict("./input/test", id_to_label_dict)	# problem: freezes program

# # 2C) NNTypeThree (epoch=1)
# nnTypeThree = NNTypeThree(verbose=VERBOSE)
# nnTypeThree.create(X_train, len(unique_labels))
# nnTypeThree.train_and_evaluate(X_train, X_test, Y_train, Y_test, epochs=1)
# nnTypeThree.predict("./input/test", id_to_label_dict)	# problem: freezes program

# # 2D) NNTypeThree_v2 (epoch=10)
nnTypeThree_v2 = NNTypeThree(verbose=VERBOSE)
nnTypeThree_v2.create(X_train, len(unique_labels))
nnTypeThree_v2.train_and_evaluate(X_train, X_test, Y_train, Y_test, epochs=10)
nnTypeThree_v2.predict("./input/test", id_to_label_dict)	# problem: freezes program
