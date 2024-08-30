import numpy as np
import cv2
import keras
from keras.layers import BatchNormalization
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential


model = Sequential()
    
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), input_shape=(100, 100, 3), activation='relu'))

# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))

# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(100, activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights('model.hdf5')
print("done loading weights of trained model")

labels = ['Anushka','Atharva','Adrija','Abhiyank','unknown']

# input image
img = cv2.imread('0022.jpg')
img1= cv2.resize(img,(256,256))
cv2.imshow('input',img)

img = cv2.resize(img, (100, 100))
img = np.reshape(img, [1, 100, 100, 3])

classes = model.predict(img)
output = np.argmax(classes)
print(labels[output])


font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img1, labels[output], (30,30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
cv2.imwrite('output.jpg',img1)
cv2.imshow('output',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()