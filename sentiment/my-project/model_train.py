import torch.optim as optim
import torch
from torch.utils.data import TensorDataset, DataLoader

# parameters
embedding_dim = 50
hid_dim = 64
final_dim = 3
max_length = 300
lr = 0.1
num_epochs = 5
batch_size=32

# load data
X_train, y_train, X_test, y_test, X_val, y_val, vocab_size = format_dataloader('sentiment/my-project/atd_separate_reduced_train_annotated.csv', 'sentiment/my-project/atd_separate_reduced_test_annotated.csv')
train_dataset = TensorDataset(X_train, torch.tensor(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, torch.tensor(y_test))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val, torch.tensor(y_val))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# get input size
input_dim = vocab_size

# other definitions
models = []
loss_fn = torch.nn.CrossEntropyLoss()
preds = [0.0 for _ in range(10)]

# model
for n in range(5):
    model = LSTM(input_dim, embedding_dim, hid_dim, final_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_model(model, train_dataloader, val_dataloader, num_epochs, optimizer, loss_fn)

    save_model(model, f"/my-project/oil_lstm_{n}.pth") # /my-project