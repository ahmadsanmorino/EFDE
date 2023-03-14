import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import librosa
import numpy as np
import os

# Define the paths to the dataset
train_dir = 'path/to/train/folder'
valid_dir = 'path/to/validation/folder'
test_dir = 'path/to/test/folder'

# Define the number of classes in the dataset
num_classes = 2 # 'engine_ok' and 'engine_fail'

# Define the input shape of the spectrograms
input_shape = (224, 224, 3) # Assuming the images are RGB and have size 224x224

# Define the batch size and number of epochs for training
batch_size = 32
epochs = 50

# Load the pre-trained model (here we use MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                               include_top=False,
                                               weights='imagenet')

# Freeze the base model layers
base_model.trainable = False

# Create a new model on top
inputs = tf.keras.Input(shape=input_shape)
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Create data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=input_shape[:2],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    target_size=input_shape[:2],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=input_shape[:2],
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=False)

# Calculate the class weights
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(train_generator.classes),
                                                  train_generator.classes)

# Define early stopping and model checkpoint callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint('engine_failure_detector.h5', save_best_only=True,
                                    monitor='val_loss', mode='min', verbose=1)

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    validation_data=valid_generator,
                    validation_steps=len(valid_generator),
                    class_weight=class_weights,
                    callbacks=[early_stop, model_checkpoint])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator,
                                     steps=len(test_generator),
                                     verbose=1)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)