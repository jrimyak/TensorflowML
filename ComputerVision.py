#Problem: Attempting to recognize different items of clothing
#Training from a dataset containing ten different types

import tensorflow as tf
print(tf.__version__)

#Callback to each a certain accuracy rather than waiting around for extra epochs
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print("\nReached 95% accuracy so cancelling training!")
        self.model.stop_training = True

#Defining a callback
callbacks=myCallback()

#Dataset is called Fashion MNIST
#Contains 70,000 item in 10 categories and each image is in 28x28 grayscale image

#Label    Description
#0        T-shirt/top
#1        Trouser
#2        Pullover
#3        Dress
#4        Coat
#5        Sandal
#6        Shirt
#7        Bag
#9        Ankle boot 

#Importing the Fash MNIST data from the 'tf.keras.datasets' API
mnist = tf.keras.datasets.fashion_mnist

#Two sets of lists: training values and test values 
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#Looking at what the values look like 
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

#Values are inegers between 0 and 255
#Easy to treat values as 0 and 1 because of a process called normalization 
#normalizing the list 
training_images  = training_images / 255.0
test_images = test_images / 255.0

#Need a training set and a testing set
#Training set trains the data to learn
#Test sets makes sure the knowledge is correct with a unseen set of data 
#Makes sure the data is not memorized, however, generalizing the data

#Designing the data
#Three layers
#Sequential: Defines a sequence of layers in the neural network 
#Flatten: images are squares so flattening the images makes then into a one-dimensional vector
#Dense: Adds a layer of neurons 

#Each layer of neurons needs an activation function 
#Using two in this one
#Relu: means if X>0 return X else return 0. Only passes values 0 or greater into the next layer of the network
#Softmax: takes a set of values and picks out the largest one
#For this data set: [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05] it will return [0,0,0,0,1,0,0,0,0] 
#Saves from needing to sort the largest value 

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
#Compiling and training the model
#Compile the model with a loss function and optimization function, then train it with the labels and data
#Goal: Figure out relationship between data and labels 
#Also using a "metrics" parameter tracks how accurate the training is going -- what's right what's wrong then reports back
model.compile(optimizer = tf.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

#Got about 89% accuracte which is okay but only trained 5 epochs and trained quickly 

#Testing the model on data it hasn't seen before with the test data and will report the loss for each
model.evaluate(test_images, test_labels)

#Getting result that it was 87.65% accurate

#Try to figure out how to improve the percentages for how to computer can regonize these patterns and learn

#Creating a set of classification for the test images
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
