from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
import numpy as np
from keras.utils import plot_model
import os
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.datasets import mnist
import keras.utils
import tensorflow as tf
from keras.preprocessing import image


def normalize_data(data):
    return tf.keras.utils.normalize(data, axis=1)


os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38/bin/'
# input = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
# output = np.array([[1], [0], [1], [0]])
batch_size = 128
number_classes = 10
epochs = 2
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, number_classes)
y_test = keras.utils.to_categorical(y_test, number_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=[5, 5], padding='valid', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy, optimizer=Adadelta(),
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

my_image = image.load_img(x_test[0])
reshaped = tf.reshape(normalize_data(my_image), [28, 28])
result = model.predict([x_test[10]])
print(result)

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
hi=tf.constant(2)

