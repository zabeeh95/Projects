import cv2
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')

# Separate features and labels
X = data.drop('0', axis=1)
y = data['0']

# Split data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

# Reshape data for the CNN model
train_X = train_x.values.reshape(train_x.shape[0], 28, 28, 1)
test_X = test_x.values.reshape(test_x.shape[0], 28, 28, 1)

# One-hot encode labels
train_yOHE = to_categorical(train_y, num_classes=26)
test_yOHE = to_categorical(test_y, num_classes=26)

# Define the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(26, activation="softmax"))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_X, train_yOHE, epochs=10, validation_data=(test_X, test_yOHE))

# Save the model
model.save('character_classifier_model.h5')

# Print accuracy and loss
print("The validation accuracy is:", history.history['val_accuracy'][-1])
print("The training accuracy is:", history.history['accuracy'][-1])
print("The validation loss is:", history.history['val_loss'][-1])
print("The training loss is:", history.history['loss'][-1])
