import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# prep the data
data_directory = "./data.csv" 
data = pd.read_csv(data_directory, header=None)
data = data.to_numpy()

# make DataSet
class ArrayDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = (self.data[idx][3:], self.data[idx][0:3])
        return sample

BATCH_SIZE = 500
training_data = ArrayDataset(data)
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print('using: ', device)

# define model
class NueralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 18),
            nn.ReLU(),
            nn.Linear(18, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
    
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
model = NueralNetwork().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(torch.float32)
        y = y.to(torch.float32)
        X, y = X.to(device), y.to(device)

        # prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# test function
def test(dataloader, model, loss_fn):
    model.eval()
    mse = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(torch.float32)
            y = y.to(torch.float32)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            mse += loss_fn(pred, y).item()
    mse /= len(dataloader)
    print(f"MSE Error: {mse:>0.3f} \n")
    return mse
# epochs = 800
mse = 10
# for t in range(epochs):
best_score = 100
while mse > 1e-2:
    # print(f"Epoch {t+1}\n-------------")
    train(train_dataloader, model, loss_fn, optimizer)
    mse = test(train_dataloader, model, loss_fn)
    if mse < best_score:
        torch.save(model.state_dict(), 'model.pth')
        best_score = mse
# save model
torch.save(model.state_dict(), 'model.pth')
print('Done!')