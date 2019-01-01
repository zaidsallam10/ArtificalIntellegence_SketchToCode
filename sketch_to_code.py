from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
import numpy as np
from keras.utils import plot_model
import os
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adadelta
from keras.datasets import mnist
import keras.utils
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# from IPython.display import display
# from PIL import Image
img_width, img_height = 64, 64


def normalize_data(data):
    return tf.keras.utils.normalize(data, axis=1)


def runModel():
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=[5, 5], padding='valid', activation='relu', input_shape=(64, 64, 3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=1, activation='softmax'))

    model = Sequential()
    123
    # Step 1 - Convolution
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    123

    # Step 2 - Pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    123

    # Step 2(b) - Add 2nd Convolution Layer making it Deep followed by a Pooling Layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    123
    model.add(MaxPooling2D(pool_size=(2, 2)))
    123

    # Step 3 - Flattening
    model.add(Flatten())
    123

    # Step 4 - Fully Connected Neural Network
    # Hidden Layer - Activation Function RELU
    model.add(Dense(units=128, activation='relu'))
    123

    # Output Layer - Activation Function Softmax(to clasify multiple classes)
    model.add(Dense(units=3, activation='softmax'))
    123

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    123

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    123

    training_set = train_datagen.flow_from_directory('C://Users/hp/Desktop/ai_python/training_cnn',
                                                     target_size=(64, 64),
                                                     batch_size=8,
                                                     class_mode='categorical')
    123
    testing_set = test_datagen.flow_from_directory('C://Users/hp/Desktop/ai_python/testing_cnn', target_size=(64, 64),
                                                   batch_size=8,
                                                   class_mode='categorical')
    123
    #
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    123

    model.fit_generator(training_set, steps_per_epoch=500, epochs=2, validation_data=testing_set,
                        validation_steps=2000,
                        shuffle=True)
    123

    # this will give the index of my classes
    print(testing_set.class_indices)
    123
    print(testing_set.classes)
    123

    # predicting images
    img = image.load_img('sketch1.png', target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    print(classes)
    return (classes)

    #  the following code for loading the model

    # loading_json_file = open('model.json', 'r')
    # loaded_json_file = loading_json_file.read()
    # loading_json_file.close()
    #
    #
    # loaded_model = model_from_json(loaded_json_file)
    # loaded_model.load_weights("model.h5")
    # print("has been loaded")
    #
    # images = np.vstack([x])
    # classes = loaded_model.predict_classes(images, batch_size=10)
    # print(classes)
    # return (classes)
