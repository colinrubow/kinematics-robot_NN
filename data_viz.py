import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_directory = "./data.csv"
data = pd.read_csv(data_directory, header=None)
data = data.to_numpy()

plt.plot(data[:,5])
plt.show()