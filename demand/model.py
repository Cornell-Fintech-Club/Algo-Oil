import torch.nn as nn
import torch
from tqdm import tqdm
import os
import pandas as pd


class FFNN(nn.Module):
    def __init__(self, input_dim, hid_dim, final_dim, num_layers):
        super().__init__()
        self.inp = nn.Linear(input_dim, hid_dim)
        self.hid = nn.ModuleList(
            [nn.Linear(hid_dim, hid_dim) for _ in range(num_layers - 1)]
            + [nn.Linear(hid_dim, final_dim)]
        )
        self.out = nn.Linear(final_dim, 1)

        self.act = nn.ReLU()

    def forward(self, input):
        x = self.inp(input)
        for hid in self.hid:
            x = self.act(hid(x))
        y = self.out(x)
        return y


def format_dataloader(file_path):
    df = pd.read_csv(file_path)
    dataloader = []
    for _, row in df.iterrows():
        # print([row["Exports"], row["field_prod"], row["imports"], row["net_imports"], row["refiner_input"], row["stocks"], row["Unemployment"], row["snp"], row["vix"], row["wti_prev"]])
        input = torch.FloatTensor(
            [
                row[
                    "U.S. Exports of Finished Petroleum Products Thousand Barrels per Day"
                ],
                row["Month"],
                row["Year"],
                row[
                    "U.S. Product Supplied of Finished Motor Gasoline (Thousand Barrels)"
                ],
                row["U.S. Electricity Consumption (billion kilowatthours per day)"],
                row["U.S. Heating Degree Days (degree days)"],
                row["U.S. Cooling Degree Days (degree days)"],
            ]
        )
        label = torch.FloatTensor(
            [row["U.S. Crude Oil Production (million barrels per day)"]]
        )
        batch = {"input": input, "label": label}
        dataloader = [batch] + dataloader
    return dataloader


def train_model(
    model, train_dataloader, val_dataloader, num_epochs, optimizer, loss_fn, file=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    validation_loss = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        total_loss = 0
        model.train()
        total_batchs = 0
        for batch in tqdm(train_dataloader, desc="Training Batches"):
            optimizer.zero_grad()
            input = batch["input"].to(device)
            label = batch["label"].to(device)

            output = model(input)
            loss = loss_fn(output, label)
            total_loss += loss
            total_batchs += 1

            loss.backward()
            optimizer.step()
        # print(f"Training Loss: {total_loss / total_batchs}")

        val_loss = 0
        total_batchs = 0
        model.eval()
        for batch in tqdm(val_dataloader, desc="validation batches"):
            input = batch["input"].to(device)
            label = batch["label"].to(device)

            output = model(input)
            loss = loss_fn(output, label)
            val_loss += loss.item()
            total_batchs += 1

        validation_loss.append(val_loss / total_batchs)
        # print(f"Validation Loss: {val_loss / total_batchs}")

    if file:
        path = os.path.join(os.getcwd(), "models", file)
        model.save_pretrained(path)

    return model, validation_loss


def eval_model(model, test_dataloader, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    total_batchs = 0
    outputs = []
    for batch in tqdm(test_dataloader, desc="test batches"):
        input = batch["input"].to(device)
        label = batch["label"].to(device)

        output = model(input)
        loss = loss_fn(output, label)
        test_loss += loss.item()
        total_batchs += 1
        outputs.append(output.item())
    print(f"Test Loss: {test_loss / total_batchs}")
    return outputs, test_loss / total_batchs


def pred_model(model, input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    input = input.to(device)

    output = model(input)
    return output


def save_model(model, path):
    torch.save(model, path)


def load_model(path):
    return torch.load(path)
