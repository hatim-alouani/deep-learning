import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selectin import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, BatchNormalization, Flatten, Dense

(x_train,x_test), (y_train , y_test) = mnist.load_data()

x_train = x_train.reshape((60000,28,28,1)).astype('float32')/255
x_test = x_test.reshape((20000,28,28,1)).astype('float32')/255

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size=0.2, random_state=42)


model = Sequential([
    Conv2D(filters=32 , kernel_size=3 ,  input_shape = (28, 28, 1) , padding='valid')
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(pool_size=2, strides=2),

    Conv2D(filters=32 , kernel_size=3 , padding='valid'),
    BatchNormalization(),
    ReLU(), 
    MaxPooling2D(pool_size=2, strides=2),

    Flatten(),
    Dense(units=128, activation='relu'),
    Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train , epochs=10, validation_data(x_test , y_test) , batch_size=32)

test_loss , test_accuracy = model.evaluate(x_test,y_test)
print("test Accuracy" , test_accuracy)
print("test loss" , test_loss)

prediction = model.predict(x_test[0:1])
prediction_label = np.argmax(prediction)
print("prediction label",prediction_label)
print("true label",y_test[0])
