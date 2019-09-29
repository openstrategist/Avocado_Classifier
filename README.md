# Basic Avocado Classifier

 - Using Keras (over tensorflow) to test image classification over ripe and non-ripe avocados.
 - Adding Apples and Collection of Apple images to classify when the object is not an avocado.
 - Dataset collected from Kaggle (Fruit 360 Dataset)
 - CNN Code based on mcbean's fruit-classification-w-nn(https://www.kaggle.com/mcbean/fruit-classification-w-nn) and shivamb's cnn-architectures-vgg-resnet-inception-tl(https://www.kaggle.com/shivamb/cnn-architectures-vgg-resnet-inception-tl)

## Future Goals
 - Detect categories of avocado ripeness.
 - Detect avocado in a normal image or a box bounded zoomed in image.

## Reference
 - https://www.kaggle.com/mcbean/fruit-classification-w-nn
 - https://www.kaggle.com/shivamb/cnn-architectures-vgg-resnet-inception-tl

## How to run?
 - Install libraries:
 	 - matplotlib
 	 - numpy
 	 - pandas
 	 - tensorflow
 	 - keras
 	 - glob
 	 - cv2 (opencv)
 - Execute "python main.py"

### Modifications
 - Change input/test, input/training and input/validation files to train your NN differently.

# TODO
 - convert conv2D NN into classes.
 - Predict function freezes the program. Need to keep running and show next figure -> uncomment step 3 of main.py once this is fixed.
 - Transfer Learning?
 - Better .gitignore
 - consider using green-level when training/data since we are only interested in avocado.
 