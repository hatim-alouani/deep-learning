import tensorflow as tf 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist

def preprocess_images(images):
    images = tf.convert_to_tensor(images)
    images = tf.image.resize(images, [224, 224])
    images = tf.image.grayscale_to_rgb(images)
    return images

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(28, 28, 1).astype('float32')/255
x_test = x_test.reshape(28, 28, 1).astype('float32')/255

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

x_train = preprocess_images(x_train)
x_test = preprocess_images(x_test)
x_val = preprocess_images(x_val)

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top = False, weights='imagenet')
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu')
    Dense(1, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, bach_size=32)