import numpy as np
import keras
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

# Define a small dataset with 2 classes: normal engine sounds and engine failure sounds
X = np.random.random((200, 224, 224, 3))
y = np.random.randint(0, 2, size=(200, 1))

# Split the dataset into training, validation, and testing sets
train_X, train_y = X[:100], y[:100]
val_X, val_y = X[100:150], y[100:150]
test_X, test_y = X[150:], y[150:]

# Set image size and batch size for training, validation, and testing data
img_size = (224, 224)
batch_size = 16

# Load pre-trained model for transfer learning
pretrained_model = MobileNetV2(input_shape=(img_size[0], img_size[1], 3), include_top=False, weights='imagenet')

# Freeze layers in pre-trained model
for layer in pretrained_model.layers:
    layer.trainable = False

# Add custom layers for engine failure detection
x = pretrained_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create fine-tuned model for engine failure detection
model = Model(inputs=pretrained_model.input, outputs=predictions)

# Compile fine-tuned model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up data generators for training, validation, and testing data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_X, train_y, batch_size=batch_size)
val_generator = val_datagen.flow(val_X, val_y, batch_size=batch_size)
test_generator = test_datagen.flow(test_X, test_y, batch_size=batch_size)

# Train the model on the training data
model.fit_generator(train_generator, steps_per_epoch=len(train_X) // batch_size, epochs=10, validation_data=val_generator, validation_steps=len(val_X) // batch_size)

# Evaluate the model on the testing data
test_loss, test_acc = model.evaluate_generator(test_generator, steps=len(test_X) // batch_size)
print('Test accuracy:', test_acc)

# Save the fine-tuned model
model.save('fine-tuned-model.h5')