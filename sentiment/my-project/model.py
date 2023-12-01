import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, LSTM, Dense, Softmax
from tensorflow.keras.models import Sequential

class LSTM(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length):
        # hid_dim, final_dim, num_layers
        super(LSTM, self).__init__()

        # embedding layer
        self.embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)

        # LSTM layer
        self.lstm_layer = LSTM(100)

        # linear layer
        self.dense_layer = Dense(units=64, activation='relu')

    def forward(self, input):
        # forward pass 
        x = self.embedding_layer(input)
        x = self.lstm_layer(x)
        x = self.dense_layer(x)

        # softmax
        output = model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # output = tf.nn.softmax(x, axis=-1)  

        return output

def format_dataloader(file_path_train, file_path_test):
    # load data
    df1 = pd.read_csv('file_path_train.csv', engine='python')
    df2 = pd.read_csv('file_path_test', engine='python')

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

    X_val = token.texts_to_sequences(X_val)
    X_val = pad_sequences(X_val, maxlen=max_length, padding='post', truncating='post')

    X_test = token.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

    return X_train, y_train, X_test, y_test, X_val, y_val, 

def train_model(
    model, train_dataloader, val_dataloader, num_epochs, optimizer, loss_fn, file=None
):

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        total_loss = 0
        total_batchs = 0

        # training
        for batch in tqdm(train_dataloader, desc="Training Batches"):
            input = batch["input"]
            label = batch["label"]

            with tf.GradientTape() as tape:
                output = model(input)
                loss = loss_fn(output, label)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            total_loss += loss.numpy()
            total_batches += 1
        
        print(f"Training Loss: {total_loss / total_batchs}")

        val_loss = 0
        total_batchs = 0

        # validation
        for batch in tqdm(val_dataloader, desc="validation batches"):
            input = batch["input"]
            label = batch["label"]

            output = model(input)            
            loss = loss_fn(output, label)
            val_loss += loss.numpy()
            total_batchs += 1

        print(f"Validation Loss: {val_loss / total_batchs}")

    if file:
        model.save(file)

    return model

def eval_model(model, test_dataloader, loss_fn):
    device = torch.device("cuda" if tf.config.experimental.list_physical_devices("GPU") else "cpu")
    model.eval()
    test_loss = 0
    total_batchs = 0
    outputs = []
    for batch in tqdm(test_dataloader, desc="test batches"):
        input = batch["input"]
        label = batch["label"]

        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        inputs, labels = inputs.to(device), labels.to(device)

        output = model(input)
        loss = loss_fn(output, label)
        test_loss += loss.numpy()
        total_batchs += 1
        outputs.extend(output.numpy())
    print(f"Test Loss: {test_loss / total_batchs}")
    return outputs, test_loss / total_batchs


def pred_model(model, input):
    device = "cuda" if tf.config.experimental.list_physical_devices("GPU") else "cpu"
    model.eval()

    input = tf.convert_to_tensor(input, dtype=tf.float32)
    input = input.to(device)

    output = model(input)
    return output.numpy()


def save_model(model, path):
    model.save(path)


def load_model(path):
    return tf.keras.models.load_model(path)

