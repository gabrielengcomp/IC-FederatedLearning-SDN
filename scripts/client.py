import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import IDSNet
import sys

CLIENT_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 2

def load_data(client_id):
    X_train = np.load(f"data/client{client_id}_X_train.npy")
    X_test  = np.load(f"data/client{client_id}_X_test.npy")
    y_train = np.load(f"data/client{client_id}_y_train.npy")
    y_test  = np.load(f"data/client{client_id}_y_test.npy")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(test_ds,  batch_size=BATCH_SIZE),
            len(train_ds), len(test_ds))

def train(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(EPOCHS):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

def evaluate(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss, correct, total = 0.0, 0, 0
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            out = model(X_batch)
            loss += criterion(out, y_batch).item()
            preds = out.argmax(1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
            tp += ((preds == 1) & (y_batch == 1)).sum().item()
            fp += ((preds == 1) & (y_batch == 0)).sum().item()
            fn += ((preds == 0) & (y_batch == 1)).sum().item()

    acc = correct / total
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return loss / len(loader), acc, precision, recall, f1

class IDSClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.model = IDSNet().to(DEVICE)
        self.trainloader, self.testloader, self.n_train, self.n_test = load_data(client_id)
        print(f"[Cliente {client_id}] {self.n_train:,} treino | {self.n_test:,} teste")

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params):
        keys = self.model.state_dict().keys()
        state = {k: torch.tensor(v) for k, v in zip(keys, params)}
        self.model.load_state_dict(state, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader)
        return self.get_parameters(config), self.n_train, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc, prec, rec, f1 = evaluate(self.model, self.testloader)
        print(f"  Loss={loss:.4f} Acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
        return loss, self.n_test, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=IDSClient(CLIENT_ID)
)