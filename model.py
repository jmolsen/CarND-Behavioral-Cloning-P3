import os
import csv

### LOAD TRAINING DATA
samples = []
training_paths = ['track1_forward', 'track1_backwards']
for training_run_path in training_paths:
    with open('./training_data/' + training_run_path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

print("len(samples)=", len(samples))

### SPLIT TRAINING DATA INTO TRAIN AND VALIDATION SAMPLES
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
from sklearn.utils import shuffle

### DEFINE CORRECTION FOR STEERING ASSOCIATED WITH SIDE CAMERA IMAGES
side_steering_correction = 0.3
BATCH=32

### METHOD USED BY GENERATOR TO APPEND DATA TO IMAGES AND ANGLES LISTS
### ALSO ADD ANY STEERING CORRECTION AND AUGMENT DATA WITH FLIPPED DATA
def appendImagesAngles(images, angles, batch_sample, sample_index=0, steering_correction=0):
    name = './training_data/'+ batch_sample[sample_index].split('/')[-3] + '/IMG/' + batch_sample[sample_index].split('/')[-1]
    image = cv2.imread(name)
    angle = float(batch_sample[3]) + steering_correction
    images.append(image)
    angles.append(angle) 
    #FLIPPED
    image_flipped = np.fliplr(image)
    angle_flipped = -angle
    images.append(image_flipped)
    angles.append(angle_flipped)

### GENERATOR FOR LOADING DATA FROM ALL THREE CAMERAS
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #CENTER
                appendImagesAngles(images, angles, batch_sample)
                #LEFT
                appendImagesAngles(images, angles, batch_sample, 1, side_steering_correction)
                #RIGHT
                appendImagesAngles(images, angles, batch_sample, 2, -side_steering_correction)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH)
validation_generator = generator(validation_samples, batch_size=BATCH)

### DEFINE NUMBER OF EPOCHS TO TRAIN
EPOCH=20
row, col, ch = 160, 320, 3
### DEFINE NUMBER OF PIXES TO CROP FROM TOP AND BOTTOM OF IMAGES
top_crop = 70
bottom_crop = 25 
### DEFINE ACTIVATION FUNCTION USED IN MODEL
activation_function = "relu"

from keras.models import Model, Sequential
from keras.layers import Cropping2D, Dense, Flatten, Lambda
from keras.layers.convolutional import Convolution2D

#NVIDIA Architecture
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((top_crop, bottom_crop), (0,0)), input_shape=(row, col, ch)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation=activation_function))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation=activation_function))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation=activation_function))
model.add(Convolution2D(64,3,3, activation=activation_function))
model.add(Convolution2D(64,3,3, activation=activation_function))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
            validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=EPOCH, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()





