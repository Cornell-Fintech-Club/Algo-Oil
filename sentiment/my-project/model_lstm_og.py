import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, LSTM, Dense, Softmax
from tensorflow.keras.models import Sequential

# load data
df1 = pd.read_csv('/Users/omishasharma/Downloads/atd_separate_reduced_annotated_train.csv', engine='python')
df2 = pd.read_csv('/Users/omishasharma/Downloads/atd_separate_reduced_annotated_train.csv', engine='python')

# split train into text and labels
X_train = df1["Title"] + " " + df1["Description"] # include both title and then description
y_train = df1["Label"] # label encoding: 0 as bullish, 1 as bearish, 2 as neutral

# split test into text and labels
X_test = df2["Title"] + " " + df2["Description"] # include both title and then description
y_test = df2["Label"] # label encoding: 0 as bullish, 1 as bearish, 2 as neutral

# split into test and val data
# X_train, other_x, y_train, other_y = train_test_split(X, Y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(other_x, other_y, test_size=0.33, random_state=42)

# tokenization of train
token = Tokenizer()
token.fit_on_texts(X_train)
X_train = token.texts_to_sequences(X_train)

# observe lengths
# Average Length: 276.97566628041716
# Median Length: 279.0
# Max Length: 548
# sequence_lengths = [len(seq) for seq in X]
# print("Average Length:", np.mean(sequence_lengths))
# print("Median Length:", np.median(sequence_lengths))
# print("Max Length:", np.max(sequence_lengths))

# standardize length
max_length = 300
X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')

# embedding layer
embedding_dim = 50
vocab_size = len(token.word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))

# LSTM layer 
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# linear layer
model.add(Dense(units=64, activation='relu'))

# softmax layer
model.add(Softmax())

# train model
X_val = token.texts_to_sequences(X_val)
X_val = pad_sequences(X_val, maxlen=max_length, padding='post', truncating='post')
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# prediction
X_test = token.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')
predictions = model.predict(X_test)

# evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# general sentiment - based on predictive probability and a threshold?
# threshold = 0.5 # change?
# predicted_sentiment = np.argmax(predictions, axis=1) if np.max(predictions, axis=1) > threshold else 2

# print("Predicted Sentiment:", predicted_sentiment)