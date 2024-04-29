import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Read data
stonk_data = pd.read_csv('/home/kodex/data/Combined_News_DJIA(train).csv', index_col='Date')

test_data = pd.read_csv('/home/kodex/data/Test_Combined_News.csv', index_col='Date')

test_data = test_data[::-1]

test_data_feat = test_data.iloc[:, 0:3]

stonk_data = stonk_data[::-1]

ending_data = stonk_data[['Open', 'High', 'Low']].tail(9)

test_data_feat = pd.concat([ending_data, test_data_feat])

target_y = stonk_data.iloc[:, 3:4]
stonk_data_feat = stonk_data.iloc[:, 0:4]

sc = StandardScaler()
stonk_data_ft = sc.fit_transform(stonk_data_feat.values)
stonk_data_ft = pd.DataFrame(columns=stonk_data_feat.columns, data=stonk_data_ft, index=stonk_data_feat.index)

test_data_ft = sc.fit_transform(test_data_feat.values)
test_data_ft = pd.DataFrame(columns=test_data_feat.columns, data=test_data_ft, index=test_data_feat.index)

target_y_ft = stonk_data_ft.iloc[:, 3:]
stonk_data_ft = stonk_data_ft.iloc[:, 0:3]

# Function to split data into sequences
def lstm_split(input_data, output_data, n_steps):
    X, y = [], []
    for i in range(len(input_data) - n_steps + 1):
        X.append(input_data[i:i + n_steps, :])
    for i in range(len(output_data) - n_steps + 1):
        y.append(output_data[i + n_steps - 1, -1])
    return np.array(X), np.array(y)

X_train, y_train = lstm_split(stonk_data_ft.values, target_y_ft.values, n_steps=10)
X_test, _ = lstm_split(test_data_ft.values, output_data=[], n_steps=10)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])
        return out

input_size = stonk_data_ft.shape[1]
hidden_size = 128
output_size = 1

model = LSTMModel(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 100
batch_size = 4

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i + batch_size]
        labels = y_train[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prediction
with torch.no_grad():
    model.eval()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_pred = model(X_test_tensor)

# Convert y_pred tensor to NumPy array
y_pred_numpy = y_pred.numpy()

# Print y_pred
print(y_pred_numpy)
