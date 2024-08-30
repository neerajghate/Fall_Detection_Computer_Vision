print("starting")
import time
import random
import keras
import numpy as np
from matplotlib import pyplot as plt

from keras.layers import BatchNormalization
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

if __name__ == '__main__':
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    session = tf.Session(config=config)
    start = time.time()
    random.seed(8675309)

    train_datagen = ImageDataGenerator( rescale=1./255,
                                        rotation_range=60,
                                        width_shift_range=0.3,
                                        height_shift_range=0.3,
                                        zoom_range=0.2,
                                        horizontal_flip=True)


    train_gen = train_datagen.flow_from_directory(
            "dataset/train/",
            target_size=(100,100),
            color_mode='rgb',
            batch_size=64,
            classes=['1','2','3','4','5'],
            class_mode='categorical'
        )

    # pre-process the data for Keras
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
    
    epochs = 10
    checkpointer = ModelCheckpoint(filepath='model.hdf5',
                                   verbose=1)
    history=model.fit_generator(train_gen,epochs=epochs,steps_per_epoch=32, callbacks=[checkpointer])

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Accuracy')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.savefig('loss')
    plt.show()
    end=time.time()

    print(end-start)
