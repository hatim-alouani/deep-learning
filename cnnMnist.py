import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
(x_train , y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,random_state = 42)


model = Sequential([
    Conv2D(filters=32, kernel_size=3, padding='VALID'),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(pool_size=2, strides=2),

    Conv2D(filters=32, kernel_size=3, padding='VALID'),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(pool_size=2, strides=2),

    Flatten(),
    Dense(units=128, activation='relu', kernel_regularizer=l2()),
    Dropout(0.3),
    Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_generator = ImageDataGenerator(10, 0.1, 0.1, 0,1).flow(x_train, y_train, batch_size=32)

model.fit(train_generator, validation_data=(x_val, y_val), epochs=10, callbacks=[EarlyStopping()])

loss_test, accuracy_test = model.evaluate(x_test, y_test)
prediction = model.predict(x_test[1:2])
prediction = np.argmax(prediction)
print("loss test : ", loss_test)
print("accuracy test : ", accuracy_test)
print("predicted result : ", prediction)
print("True result : ", y_test[1])