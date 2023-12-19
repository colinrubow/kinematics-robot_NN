import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

data_files = ['./data_'+str(i)+'.csv' for i in range(4)]

# prep the data
data = [pd.read_csv(d_f, header=None).to_numpy() for d_f in data_files]

device = "cpu"

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

# load the model
model = NueralNetwork()
model.load_state_dict(torch.load('./model_dense.pth'))
model.eval()

loss_fn = nn.MSELoss()

def get_mse_distribution(data, model, loss_fn):
    mse = []
    with torch.no_grad():
        for d in data:
            X, Y = d[3:], d[0:3]
            X = torch.tensor(X, dtype=torch.float32)
            Y = torch.tensor(Y, dtype=torch.float32)
            pred = model(X)
            mse.append(loss_fn(pred, Y))
    return mse

mses = [get_mse_distribution(d, model, loss_fn) for d in data]

plt.boxplot(mses, sym="")
plt.ylabel('MSE(rad)')
plt.xlabel('Trial #')
plt.show()