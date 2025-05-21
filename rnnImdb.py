import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)

def simple_rnn():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=100))
    model.add(tf.keras.layers.SimpleRNN(32, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=0)
    return model.evaluate(x_test, y_test) 

def bidirectional_rnn():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=100))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(32, activation='tanh')))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=0)
    return model.evaluate(x_test, y_test) 

def lstm():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=100))
    model.add(tf.keras.layers.LSTM(32, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=0)
    return model.evaluate(x_test, y_test)

def bidirectional_lstm():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=100))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='tanh')))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=0)
    return model.evaluate(x_test, y_test)

def gru():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=100))
    model.add(tf.keras.layers.GRU(32, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=0)
    return model.evaluate(x_test, y_test)


def bidirectional_gru():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=100))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, activation='tanh')))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=0)
    return model.evaluate(x_test, y_test)


test_loss, test_accuracy = simple_rnn()
print("Test Accuracy (Simple RNN):", test_accuracy)
print("Test Loss (Simple RNN):", test_loss)
test_loss, test_accuracy = bidirectional_rnn()
print("Test Accuracy (Bidirectional RNN):", test_accuracy)
print("Test Loss (Bidirectional RNN):", test_loss)
test_loss, test_accuracy = lstm()
print("Test Accuracy (LSTM):", test_accuracy)
print("Test Loss (LSTM):", test_loss)
test_loss, test_accuracy = bidirectional_lstm()
print("Test Accuracy (Bidirectional LSTM):", test_accuracy)
print("Test Loss (Bidirectional LSTM):", test_loss)
test_loss, test_accuracy = gru()
print("Test Accuracy (GRU):", test_accuracy)
print("Test Loss (GRU):", test_loss)
test_loss, test_accuracy = bidirectional_gru()
print("Test Accuracy (Bidirectional GRU):", test_accuracy)
print("Test Loss (Bidirectional GRU):", test_loss)

