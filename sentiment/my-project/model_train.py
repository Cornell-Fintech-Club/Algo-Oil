from model import (
    LSTM,
    format_dataloader,
    train_model,
    pred_model,
    save_model,
    load_model,
)

# parameters
embedding_dim = 50
max_length = 300
lr = 0.1

# load data
df1 = pd.read_csv('/Users/omishasharma/Downloads/atd_separate_reduced_annotated_train.csv', engine='python')
df2 = pd.read_csv('/Users/omishasharma/Downloads/atd_separate_reduced_annotated_train.csv', engine='python')

models = []
loss_fn = tf.keras.losses.CategoricalCrossentropy()
preds = [0.0 for _ in range(10)]

for n in range(5):
    model = LSTM(vocab_size, embedding_dim, max_length)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    # might need extra step to convert the X/y split to one data frame for both train & test & val?
    train_model(model, train + val, test, num_epochs, optimizer, loss_fn)

    save_model(model, f"models/oil_ffnn_{n}.pth")