import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, GRU, LSTM, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

x_train = pad_sequences(x_train, 100)
x_test = pad_sequences(x_test, 100)
x_val = pad_sequences(x_val, 100)

model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),
    SimpleRNN(32, activation='tanh'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(units=1, activation='sigmoid', kernel_regularizer=l2())
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=64, callbacks=[EarlyStopping()])
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(test_accuracy)
print(test_loss)
prediction = model.predict(x_test[0:1])
print(prediction)
print(y_test[0])
