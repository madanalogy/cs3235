import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import util

data = util.load()
seq_len = 0
for idx, seq in data.iterrows():
    length, cols = seq['sequence'].shape
    if length > seq_len:
        seq_len = length
print(seq_len)
feat_len = 3

X_train, X_test, y_train, y_test = train_test_split(data['sequence'], data['url'], test_size=0.1)
print(X_train.shape)
X_train = sequence.pad_sequences(X_train, maxlen=seq_len)
X_test = sequence.pad_sequences(X_test, maxlen=seq_len)
print(X_train.shape)

model = keras.Sequential()
model.add(layers.LSTM(128, input_shape=(seq_len, feat_len), return_sequences=False))
model.add(layers.Dense(35))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.3f%%" % (scores[1]*100))
