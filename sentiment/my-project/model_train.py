import torch.optim as optim
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

# parameters
embedding_dim = 64
hid_dim = 64
final_dim = 3
max_length = 150 #300
lr = 0.001 #0.001
num_epochs = 5
batch_size= 16

# load data
X_train, y_train, X_test, y_test, X_val, y_val, vocab_size = format_dataloader('sentiment/my-project/atd_new_train_annotated.csv', 'sentiment/my-project/atd_new_test_annotated.csv')
train_dataset = TensorDataset(X_train, torch.tensor(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, torch.tensor(y_test))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val, torch.tensor(y_val))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# get input size
input_dim = vocab_size


# other definitions
# models = []
loss_fn = torch.nn.CrossEntropyLoss() # CrossEntropyLoss
# loss_fn = torch.nn.NLLLoss() # NLLLoss
# preds = [0.0 for _ in range(10)]

# model
for n in range(1):
    model = LSTM(input_dim, embedding_dim, hid_dim, final_dim, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_epochs, train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, num_epochs, optimizer, loss_fn)

    # Plotting the training loss over epochs
    plt.plot(range(1, num_epochs + 1), train_losses, color="red", label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, color="blue", label='Validation Loss')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    save_model(model, f"my-project/oil_lstm_{n}.pth") # /my-project

preds = []
labels = []
for batch in test_dataloader:
  out = pred_model(model, batch[0])
  out_labels = torch.argmax(out, dim=1)
  preds += out_labels.tolist()
  labels += batch[1].tolist()

# calc f1
precision, recall, F1 = {}, {}, {}
reverse_mapping = {0: "Bullish", 1: "Bearish", 2: "Neutral"}

for k in reverse_mapping:
  label = reverse_mapping[k]
  correct = 0
  false_pos = 0
  false_neg = 0
  for i in range(len(preds)):
    if preds[i] == labels[i] and preds[i] == k:
      correct += 1
    elif preds[i] != labels[i] and preds[i] == k:
      false_pos += 1
    elif preds[i] != labels[i] and labels[i] == k:
      false_neg += 1

  if correct == 0:
    precision[label] = 0
    recall[label] = 0
    F1[label] = 0
  else:
    precision[label] = correct / (correct + false_pos)
    recall[label] = correct / (correct + false_neg)
    F1[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label])

macro_avg = {"Precision": 0, "Recall": 0, "F1": 0}
for k in reverse_mapping:
  label = reverse_mapping[k]
  macro_avg["Precision"] += precision[label] / len(reverse_mapping)
  macro_avg["Recall"] += recall[label] / len(reverse_mapping)
  macro_avg["F1"] += F1[label] / len(reverse_mapping)

print("Precision")
print(precision)
print("Recall")
print(recall)
print("F1")
print(F1)
print("Macro AVG")
print(macro_avg)