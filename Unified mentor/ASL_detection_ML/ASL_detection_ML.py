import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

train_dir = '/home/shobhit/AI-ML/Unified mentor/ASL_detection_ML/archive/asl_alphabet_train/asl_alphabet_train'
test_dir = '/home/shobhit/AI-ML/Unified mentor/ASL_detection_ML/archive/asl_alphabet_test/asl_alphabet_test'
#loading training and testing data

img_height = 200
img_width = 200
batch_size = 32


#here we will create ImageDataGenerator for data Augmentation 
#this is not necessary if data is huge enough but still if we want to make it diverse
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2)

test_datagen = ImageDataGenerator(
    rescale = 1./255
    #we just need to do this in this
    #because 
)

#create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width), 
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'#use the training subset
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

#model creation

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(28, activation='softmax') #28 classes for ASL cause P excluded
])

#here we are compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#here we are training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs = 5,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size
)

#here we are saving the model
model.save('asl_model.h5')

#evaluation of the model 

#plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and validation Loss')

plt.savefig('training_history.png')
plt.show()