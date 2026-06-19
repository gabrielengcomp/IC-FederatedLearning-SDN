import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST(".", train=True,  download=True, transform=tf)
    testset  = datasets.MNIST(".", train=False, download=True, transform=tf)
    return DataLoader(trainset, batch_size=32, shuffle=True),            DataLoader(testset,  batch_size=32)

def train(model, loader, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

def evaluate(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return loss / len(loader), correct / len(loader.dataset)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net().to(DEVICE)
        self.trainloader, self.testloader = load_data()

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params):
        keys = self.model.state_dict().keys()
        state = {k: torch.tensor(v) for k, v in zip(keys, params)}
        self.model.load_state_dict(state, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader)
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = evaluate(self.model, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": acc}

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)