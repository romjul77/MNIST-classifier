import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Image enhancement and preprocessing
def enhance_image(image):
    enhanced_image = cv2.equalizeHist(image)
    return enhanced_image

# Image segmentation (if needed)
def segment_image(image):
    _, segmented_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return segmented_image

# Preprocess the dataset
x_train_processed = np.array([enhance_image(img) for img in x_train])
x_test_processed = np.array([enhance_image(img) for img in x_test])

# Flatten the images and normalize pixel values
x_train_processed = x_train_processed.reshape((-1, 28, 28, 1)) / 255.0
x_test_processed = x_test_processed.reshape((-1, 28, 28, 1)) / 255.0

# One-hot encode the labels
y_train_one_hot = to_categorical(y_train, num_classes=10)
y_test_one_hot = to_categorical(y_test, num_classes=10)

# Build a simple neural network model for digit classification
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train_processed, y_train_one_hot, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test_processed, y_test_one_hot)
print(f'Test Accuracy: {test_accuracy}')

# Make predictions on a sample image (you can use your own test image)
sample_image = x_test_processed[0].reshape((1, 28, 28, 1))
predictions = model.predict(sample_image)
predicted_digit = np.argmax(predictions)

print(f'Predicted Digit: {predicted_digit}')
